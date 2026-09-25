"""
Objective Multi-Turn Benchmark Evaluation Pipeline.

Structures metrics into 3 core pillars:
A. TOOL USE
   - tool_decision_accuracy
   - tool_argument_accuracy
   - tool_switching_accuracy
   - missing_tool_call_rate
   - unnecessary_tool_call_rate

B. MULTI-TURN
   - context_slot_accuracy
   - coreference_accuracy
   - follow_up_understanding
   - state_tracking_accuracy
   - cross_turn_consistency

C. ANSWER QUALITY
   - result_summarization_score
   - final_answer_quality
   - grounded_claim_precision
   - unsupported_claim_rate
   - contradiction_rate

Overall:
   - task_success_rate

Usage:
    python -m eval.tool_call.scripts.run_trajectory_eval --model gemma4:e2b
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from eval.tool_call.scripts.enrich_mentor_multiturn import enrich_batch_dataset
except ImportError:
    enrich_batch_dataset = None


# ==============================================================================
# Helper: Atomic Claim Extraction & Cleaning
# ==============================================================================

def extract_clean_claims(text: str) -> List[str]:
    """Extracts atomic semantic claims, filtering out markdown headers, boilerplate, and stripping introductory clauses."""
    lines = [s.strip() for s in re.split(r'[\n;]+|(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    claims = []
    prefix_pattern = re.compile(
        r'^(dựa trên thông tin.*?,\s*|'
        r'dựa vào thông tin.*?,\s*|'
        r'theo thông tin.*?,\s*|'
        r'theo như thông tin.*?,\s*|'
        r'dưới đây là.*?[:\n]\s*|'
        r'tóm lại,?\s*|'
        r'như vậy,?\s*)',
        re.IGNORECASE
    )
    for line in lines:
        cleaned = prefix_pattern.sub('', line).strip()
        cleaned = re.sub(r'^[#*\-\d.\s]+', '', cleaned).strip()
        cleaned = re.sub(r'^\*\*.*?\*\*:?', '', cleaned).strip()
        cleaned = cleaned.strip('*').strip()
        if not cleaned or len(cleaned) < 15:
            continue
        if cleaned.endswith(':') or cleaned.startswith('**'):
            continue
        if any(cleaned.lower().startswith(b) for b in [
            'rất vui được hỗ trợ', 'tôi có thể giúp', 'nếu bạn cần thêm thông tin'
        ]):
            continue
        claims.append(cleaned)
    return claims


# ==============================================================================
# 1. Deterministic Rule Validator (Zero-LLM)
# ==============================================================================

class DeterministicValidator:
    """Validates tool decisions, arguments, slot retention, and coreferences using deterministic rules."""

    @staticmethod
    def evaluate_turn_tools(turn_data: Dict[str, Any], turn_gt: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates tool decision, tool name, and query argument quality."""
        agent_resp = turn_data.get("agent_response", {})
        actual_called = (agent_resp.get("type") == "tool_call")
        
        gt_tool = turn_gt.get("tool_call", {})
        should_call = gt_tool.get("should_call", False)
        expected_name = gt_tool.get("expected_name")
        required_keywords = [kw.lower() for kw in gt_tool.get("required_arg_keywords", [])]

        tool_decision_accuracy = 1.0 if (actual_called == should_call) else 0.0
        missing_tool_call = (should_call and not actual_called)
        unnecessary_tool_call = (not should_call and actual_called)

        tool_name_accuracy = 1.0
        tool_argument_accuracy = 1.0
        actual_tool_name = None
        actual_args = {}

        if actual_called:
            tc = agent_resp.get("tool_call", {})
            actual_tool_name = tc.get("name")
            actual_args = tc.get("arguments", {})
            if expected_name:
                tool_name_accuracy = 1.0 if (actual_tool_name == expected_name) else 0.0

            if required_keywords:
                raw_query_val = actual_args.get("query") or actual_args.get("queries") or actual_args
                query_str = str(raw_query_val).lower()
                matched_kw = [kw for kw in required_keywords if kw in query_str]
                tool_argument_accuracy = len(matched_kw) / len(required_keywords)

        return {
            "actual_called_tool": actual_called,
            "should_call_tool": should_call,
            "tool_decision_accuracy": tool_decision_accuracy,
            "missing_tool_call": missing_tool_call,
            "unnecessary_tool_call": unnecessary_tool_call,
            "tool_name_accuracy": tool_name_accuracy if actual_called else None,
            "tool_argument_accuracy": tool_argument_accuracy if actual_called else None,
            "actual_tool_name": actual_tool_name,
            "actual_args": actual_args
        }

    @staticmethod
    def evaluate_context_slots(answer_text: str, turn_gt: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates context slot retention (exact / alias match in answer)."""
        required_slots = turn_gt.get("required_slots", {})
        if not required_slots:
            return {
                "context_slot_accuracy": 1.0,
                "slots_evaluated": {},
                "total_required_slots": 0,
                "matched_slots_count": 0
            }

        norm_answer = answer_text.lower()
        slots_evaluated = {}
        matched_count = 0

        for slot_name, aliases in required_slots.items():
            matched_alias = None
            for alias in aliases:
                alias_clean = alias.strip().lower()
                if alias_clean in norm_answer:
                    matched_alias = alias
                    break
            is_matched = (matched_alias is not None)
            if is_matched:
                matched_count += 1
            slots_evaluated[slot_name] = {
                "matched": is_matched,
                "matched_alias": matched_alias,
                "expected_aliases": aliases
            }

        acc = matched_count / len(required_slots) if required_slots else 1.0
        return {
            "context_slot_accuracy": acc,
            "slots_evaluated": slots_evaluated,
            "total_required_slots": len(required_slots),
            "matched_slots_count": matched_count
        }

    @staticmethod
    def evaluate_coreference(
        answer_text: str,
        turn_gt: Dict[str, Any],
        slot_eval: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluates coreference resolution via explicit target mention or contextual slot grounding."""
        coref_gt = turn_gt.get("coreference")
        if not coref_gt:
            return {
                "coreference_accuracy": None,
                "has_coreference": False
            }

        expected_targets = [t.strip().lower() for t in coref_gt.get("expected_target", [])]
        norm_answer = answer_text.lower()
        matched_target = None
        for target in expected_targets:
            if target in norm_answer:
                matched_target = target
                break

        explicit_matched = (matched_target is not None)

        # Context-grounded check: If target antecedent is not explicitly mentioned,
        # check whether the model correctly resolved context by answering the turn's required slots
        context_grounded = False
        if not explicit_matched and slot_eval:
            slot_acc = slot_eval.get("context_slot_accuracy", 0.0)
            matched_count = slot_eval.get("matched_slots_count", 0)
            if slot_acc >= 0.5 or matched_count > 0:
                context_grounded = True

        is_resolved = explicit_matched or context_grounded
        resolution_mode = "explicit_mention" if explicit_matched else ("context_grounded" if context_grounded else "unresolved")

        return {
            "has_coreference": True,
            "referent_phrase": coref_gt.get("referent_phrase"),
            "expected_target": coref_gt.get("expected_target"),
            "matched_target": matched_target,
            "resolution_mode": resolution_mode,
            "context_grounded": context_grounded,
            "coreference_accuracy": 1.0 if is_resolved else 0.0
        }


# ==============================================================================
# 2. Atomic Claim Verifier & Groundedness (BINEVAL / FactScore style)
# ==============================================================================

class AtomicClaimVerifier:
    """Decomposes response into atomic claims and verifies grounding against evidence using MODEL (e.g. gemma-4-31B-it)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        enabled: bool = True
    ):
        self.base_url = (
            base_url
            or os.environ.get("BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://more-starring-administration-cemetery.trycloudflare.com/v1"
        )
        self.model = (
            model
            or os.environ.get("MODEL")
            or os.environ.get("LLM_MODEL")
            or "gemma-4-31B-it"
        )
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or "EMPTY"
        )
        self.client = None
        if enabled and self.base_url:
            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )

    async def verify_claims(
        self,
        answer_text: str,
        evidence_text: str,
        user_message: str
    ) -> Dict[str, Any]:
        """Decomposes answer into atomic claims and labels each as SUPPORTED, CONTRADICTED, or UNSUPPORTED."""
        if not answer_text.strip():
            return {
                "claims": [],
                "grounded_claim_precision": 0.0,
                "unsupported_claim_rate": 0.0,
                "contradiction_rate": 0.0,
                "total_claims": 0
            }

        if not self.client:
            return self._heuristic_fallback(answer_text, evidence_text)

        prompt = f"""Bạn là hệ thống kiểm chứng tính chân thực (Atomic Claim & Grounding Verifier) theo chuẩn FactScore.

MỤC TIÊU:
Phân rã câu trả lời của Trợ lý thành các phát biểu đơn nguyên tử (atomic claims) và kiểm chứng từng phát biểu đối chiếu với Bằng chứng (Evidence) theo 3 nhãn: SUPPORTED, CONTRADICTED, UNSUPPORTED.

BẰNG CHỨNG (EVIDENCE TỪ TOOL VÀ NGỮ CẢNH HỘI THOẠI):
\"\"\"
{evidence_text[:4000]}
\"\"\"

CÂU HỎI CỦA NGƯỜI DÙNG:
\"{user_message}\"

CÂU TRẢ LỜI CỦA TRỢ LÝ CẦN KIỂM CHỨNG:
\"\"\"
{answer_text}
\"\"\"

HƯỚNG DẪN KIỂM CHỨNG:
1. Phân rã câu trả lời thành danh sách các phát biểu đơn nguyên tử (atomic claims). Bỏ qua các tiêu đề markdown, câu chào hỏi hoặc mở đầu xã giao.
2. Đối chiếu từng phát biểu với BẰNG CHỨNG ở trên và gán một trong ba nhãn:
   - "SUPPORTED": Bằng chứng khẳng định rõ ràng hoặc suy diễn tất yếu trực tiếp từ bằng chứng.
   - "CONTRADICTED": Bằng chứng khẳng định điều trái ngược hoặc mâu thuẫn với phát biểu.
   - "UNSUPPORTED": Bằng chứng KHÔNG HỀ ĐỀ CẬP hoặc không đủ căn cứ để kết luận (ảo giác/suy đoán ngoài văn bản).
3. Nếu nhãn là "SUPPORTED", ghi rõ trích dẫn nguyên văn từ bằng chứng (evidence_quote). Nếu "UNSUPPORTED" hoặc "CONTRADICTED", để trống hoặc trích dẫn câu mâu thuẫn.
4. Ghi rõ giải thích ngắn gọn lý do gán nhãn trong "reason".

Trả về CHỈ DUY NHẤT một chuỗi JSON hợp lệ theo format:
{{
  "claims": [
    {{
      "claim": "Chuỗi phát biểu đơn",
      "label": "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED",
      "evidence_quote": "Trích dẫn nguyên văn câu từ Bằng chứng (để trống nếu UNSUPPORTED)",
      "reason": "Giải thích ngắn gọn lý do gán nhãn"
    }}
  ]
}}
"""

        try:
            res = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_text = res.choices[0].message.content.strip()
            clean_text = re.sub(r'<thought>[\s\S]*?</thought>', '', raw_text, flags=re.IGNORECASE).strip()
            clean_text = re.sub(r'^```(?:json)?\n?', '', clean_text)
            clean_text = re.sub(r'\n?```$', '', clean_text)
            match = re.search(r'\{[\s\S]*\}', clean_text)
            parsed = json.loads(match.group()) if match else json.loads(clean_text)
            claims = parsed.get("claims", [])
            return self._compute_claim_metrics(claims)

        except Exception as e:
            print(f"⚠️ [AtomicClaimVerifier] Warning: LLM claim verification with {self.model} failed ({e}), falling back to heuristic.")
            return self._heuristic_fallback(answer_text, evidence_text)

    def _compute_claim_metrics(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(claims)
        if total == 0:
            return {
                "claims": [],
                "grounded_claim_precision": None,
                "unsupported_claim_rate": 0.0,
                "contradiction_rate": 0.0,
                "total_claims": 0,
                "supported_claims_count": 0,
                "unsupported_claims_count": 0,
                "contradicted_claims_count": 0
            }

        # Normalize label to standard 3-class uppercase
        for c in claims:
            lbl = str(c.get("label", "UNSUPPORTED")).strip().upper()
            if lbl not in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED"):
                lbl = "UNSUPPORTED"
            c["label"] = lbl

        supported = sum(1 for c in claims if c.get("label") == "SUPPORTED")
        unsupported = sum(1 for c in claims if c.get("label") == "UNSUPPORTED")
        contradicted = sum(1 for c in claims if c.get("label") == "CONTRADICTED")

        return {
            "claims": claims,
            "grounded_claim_precision": round(supported / total, 3),
            "unsupported_claim_rate": round(unsupported / total, 3),
            "contradiction_rate": round(contradicted / total, 3),
            "total_claims": total,
            "supported_claims_count": supported,
            "unsupported_claims_count": unsupported,
            "contradicted_claims_count": contradicted
        }

    def _extract_evidence_quote(self, claim: str, evidence_text: str) -> str:
        """
        Extract the sentence from evidence that best supports the claim.
        Uses substring matching to find the sentence in evidence that matches the claim.
        """
        if not claim or not evidence_text:
            return ""

        # Split evidence into sentences
        evidence_sentences = re.split(r'[.!?\n]+', evidence_text)
        norm_claim = claim.lower()

        best_match = ""
        best_ratio = 0.0

        for sent in evidence_sentences:
            sent_clean = sent.strip()
            if len(sent_clean) < 10:
                continue

            # Calculate word overlap between claim and this sentence
            claim_words = set(w.lower() for w in re.findall(r'\w+', norm_claim) if len(w) > 2)
            sent_words = set(w.lower() for w in re.findall(r'\w+', sent_clean.lower()))

            if not claim_words:
                continue

            overlap = len(claim_words & sent_words)
            ratio = overlap / len(claim_words)

            if ratio > best_ratio and ratio >= 0.3:
                best_ratio = ratio
                best_match = sent_clean

        return best_match if best_ratio >= 0.3 else ""

    def _heuristic_fallback(self, answer_text: str, evidence_text: str) -> Dict[str, Any]:
        """
        Refined sentence overlap fallback with header filtering and 3-class schema.
        """
        clean_sentences = extract_clean_claims(answer_text)
        claims = []
        norm_ev = evidence_text.lower()

        for s in clean_sentences:
            words = [w for w in re.findall(r'\w+', s.lower()) if len(w) > 2]
            overlap = sum(1 for w in words if w in norm_ev)
            ratio = overlap / max(len(words), 1)

            # Extract evidence quote if there's enough overlap
            evidence_quote = self._extract_evidence_quote(s, evidence_text)

            if ratio >= 0.55 and evidence_quote:
                label = "SUPPORTED"
                reason = f"Overlap ratio: {ratio:.2f}"
            else:
                label = "UNSUPPORTED"
                if ratio >= 0.55 and not evidence_quote:
                    reason = f"Overlap ratio: {ratio:.2f} (no evidence quote found)"
                else:
                    reason = f"Overlap ratio: {ratio:.2f}"

            claims.append({
                "claim": s,
                "label": label,
                "evidence_quote": evidence_quote,
                "reason": reason
            })
        return self._compute_claim_metrics(claims)


# ==============================================================================
# 2B. BinEval Aspect-Based Binary Evaluator (Checklist-based Rubrics)
# ==============================================================================

class BinEvalEvaluator:
    """
    Implements BinEval (Binary Evaluation) by decomposing evaluation criteria into
    an objective checklist of YES/NO questions across 3 core aspects:
      1. Tool Proficiency & Action
      2. Multi-Turn Context & Continuity
      3. Factuality & Response Quality
    """

    @staticmethod
    def evaluate_turn(
        turn_data: Dict[str, Any],
        turn_gt: Dict[str, Any],
        tool_eval: Dict[str, Any],
        slot_eval: Dict[str, Any],
        coref_eval: Dict[str, Any],
        claim_eval: Dict[str, Any],
        final_ans_quality: float
    ) -> Dict[str, Any]:
        """Evaluates a single conversation turn across 3 binary-question aspects."""

        # --- Aspect 1: Tool Proficiency ---
        tool_questions = []
        # Q1.1: Decision accuracy
        decision_pass = (tool_eval.get("tool_decision_accuracy", 0.0) == 1.0)
        tool_questions.append({
            "id": "tool_decision_pass",
            "question": "Quyết định gọi tool (hoặc trả lời trực tiếp) có chính xác theo yêu cầu ngữ cảnh không?",
            "verdict": "YES" if decision_pass else "NO",
            "reason": "Quyết định gọi tool trùng khớp với Ground Truth" if decision_pass else "Bỏ sót gọi tool hoặc gọi tool không cần thiết"
        })

        # Q1.2: Query quality (if tool was called)
        if tool_eval.get("actual_called_tool"):
            arg_acc = tool_eval.get("tool_argument_accuracy")
            query_pass = (arg_acc is not None and arg_acc >= 0.5)
            tool_questions.append({
                "id": "tool_query_pass",
                "question": "Tham số gọi tool (query) có chứa các từ khóa trọng tâm cần thiết không?",
                "verdict": "YES" if query_pass else "NO",
                "reason": f"Độ chính xác từ khóa query đạt {arg_acc*100:.0f}%" if arg_acc is not None else "Không có tham số query hợp lệ"
            })

        # Q1.3: Efficiency / Non-redundancy
        unnecessary = tool_eval.get("unnecessary_tool_call", False)
        tool_questions.append({
            "id": "tool_efficiency_pass",
            "question": "Agent có tránh được việc gọi lặp hoặc gọi công cụ dư thừa không?",
            "verdict": "NO" if unnecessary else "YES",
            "reason": "Gọi dư thừa tool khi không có yêu cầu" if unnecessary else "Hành động gọi tool tinh gọn, không dư thừa"
        })

        tool_score = round(sum(1 for q in tool_questions if q["verdict"] == "YES") / len(tool_questions), 3) if tool_questions else 1.0

        # --- Aspect 2: Multi-Turn Context & Continuity ---
        context_questions = []
        # Q2.1: Coreference resolution (if turn has coreference)
        if coref_eval.get("has_coreference"):
            coref_pass = (coref_eval.get("coreference_accuracy", 0.0) == 1.0)
            context_questions.append({
                "id": "coreference_pass",
                "question": "Agent có giải quyết đúng thực thể được ám chỉ bởi đại từ/từ thay thế không?",
                "verdict": "YES" if coref_pass else "NO",
                "reason": f"Bắt đúng đối tượng '{coref_eval.get('matched_target')}'" if coref_pass else "Không nhận diện được đối tượng tham chiếu"
            })

        # Q2.2: Context slot coverage
        total_slots = slot_eval.get("total_required_slots", 0)
        slot_acc = slot_eval.get("context_slot_accuracy", 1.0)
        if total_slots > 0:
            slot_pass = (slot_acc >= 0.5)
            context_questions.append({
                "id": "slot_coverage_pass",
                "question": "Câu trả lời có bao phủ các thông tin/thực thể then chốt của câu hỏi không?",
                "verdict": "YES" if slot_pass else "NO",
                "reason": f"Độ bao phủ thực thể đạt {slot_acc*100:.0f}% ({slot_eval.get('matched_slots_count')}/{total_slots} slots)"
            })
        else:
            context_questions.append({
                "id": "slot_coverage_pass",
                "question": "Câu trả lời có bám sát thông tin ngữ cảnh được yêu cầu không?",
                "verdict": "YES",
                "reason": "Lượt này không có ràng buộc slot đặc thù"
            })

        context_score = round(sum(1 for q in context_questions if q["verdict"] == "YES") / len(context_questions), 3) if context_questions else 1.0

        # --- Aspect 3: Factuality & Quality ---
        fact_questions = []
        # Q3.1: Grounded facts
        ground_prec = claim_eval.get("grounded_claim_precision")
        ground_pass = (ground_prec is not None and ground_prec >= 0.70) if ground_prec is not None else True
        supported_quotes = [
            c.get("evidence_quote") for c in claim_eval.get("claims", [])
            if c.get("label") == "SUPPORTED" and c.get("evidence_quote")
        ]
        reason_q1 = f"Tỷ lệ luận điểm có bằng chứng đạt {ground_prec*100:.0f}%" if ground_prec is not None else "Không có claim phát sinh"
        if supported_quotes:
            sample_quote = supported_quotes[0][:90] + ("..." if len(supported_quotes[0]) > 90 else "")
            reason_q1 += f" | Bằng chứng: \"{sample_quote}\""

        fact_questions.append({
            "id": "grounding_pass",
            "question": "Phần lớn các dữ kiện trong câu trả lời có căn cứ xác thực từ tài liệu không?",
            "verdict": "YES" if ground_pass else "NO",
            "reason": reason_q1
        })

        # Q3.2: No contradiction / Hallucination-free
        contra_rate = claim_eval.get("contradiction_rate", 0.0)
        no_contra_pass = (contra_rate == 0.0)
        fact_questions.append({
            "id": "no_contradiction_pass",
            "question": "Câu trả lời có hoàn toàn không chứa phát biểu nào mâu thuẫn/sai lệch với bằng chứng không?",
            "verdict": "YES" if no_contra_pass else "NO",
            "reason": "Hoàn toàn không có phát biểu mâu thuẫn" if no_contra_pass else f"Phát hiện {claim_eval.get('contradicted_claims_count', 1)} phát biểu mâu thuẫn ({contra_rate*100:.1f}%)"
        })

        # Q3.3: Answer completeness & structure
        completeness_pass = (final_ans_quality >= 0.65)
        fact_questions.append({
            "id": "completeness_pass",
            "question": "Câu trả lời có giải quyết trọn vẹn yêu cầu của người dùng một cách mạch lạc không?",
            "verdict": "YES" if completeness_pass else "NO",
            "reason": f"Chất lượng câu trả lời tổng thể đạt {final_ans_quality*100:.0f}%"
        })

        fact_score = round(sum(1 for q in fact_questions if q["verdict"] == "YES") / len(fact_questions), 3) if fact_questions else 1.0

        # Overall Turn BinEval score (average of aspects)
        turn_bineval_score = round((tool_score + context_score + fact_score) / 3.0, 3)

        return {
            "aspects": {
                "tool_proficiency": {
                    "score": tool_score,
                    "questions": tool_questions
                },
                "context_continuity": {
                    "score": context_score,
                    "questions": context_questions
                },
                "factuality_quality": {
                    "score": fact_score,
                    "questions": fact_questions
                }
            },
            "turn_score": turn_bineval_score
        }


# ==============================================================================
# 3. Trajectory Evaluator Engine with 3-Pillar Taxonomy & BinEval
# ==============================================================================

class TrajectoryEvaluator:
    """Full pipeline evaluating trajectory against structured Ground Truth across 3 pillars and BinEval."""

    def __init__(self, claim_verifier: Optional[AtomicClaimVerifier] = None):
        self.validator = DeterministicValidator()
        self.verifier = claim_verifier or AtomicClaimVerifier()
        self.bineval = BinEvalEvaluator()

    async def evaluate_scenario(
        self,
        trajectory: Dict[str, Any],
        scenario_gt: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluates an entire multi-turn scenario."""
        sample_id = trajectory.get("sample_id")
        turns = trajectory.get("turns", [])
        
        gt_turns = scenario_gt.get("scenario", {}).get("turns", [])
        gt_lookup = {t.get("turn"): t.get("ground_truth", {}) for t in gt_turns}
        
        source_data = scenario_gt.get("source", {})
        article_text = source_data.get("text", "") if isinstance(source_data, dict) else scenario_gt.get("article_text", "")

        evaluated_turns = []
        cumulative_context = f"Nội dung bài báo gốc:\n{article_text}\n"
        prior_answers = []

        for turn_data in turns:
            turn_num = turn_data.get("turn")
            user_msg = turn_data.get("user_message", "")
            final_ans = turn_data.get("final_answer", "")
            turn_gt = gt_lookup.get(turn_num, {})

            # 1. Tool Evaluation
            tool_eval = self.validator.evaluate_turn_tools(turn_data, turn_gt)

            # 2. Slot Tracking
            slot_eval = self.validator.evaluate_context_slots(final_ans, turn_gt)

            # 3. Coreference
            coref_eval = self.validator.evaluate_coreference(final_ans, turn_gt, slot_eval=slot_eval)

            # Evidence update
            tool_res = turn_data.get("tool_result")
            if tool_res and isinstance(tool_res, dict):
                results = tool_res.get("results", [])
                for r in results:
                    snippet = r.get("snippet", "")
                    if snippet and snippet not in cumulative_context:
                        cumulative_context += f"\nKết quả công cụ web_search:\n{snippet}\n"

            # 4. Atomic Claim Verification
            claim_eval = await self.verifier.verify_claims(
                answer_text=final_ans,
                evidence_text=cumulative_context,
                user_message=user_msg
            )

            # 5. Turn-level quality metrics:
            # - Result summarization score (for turns where tool was called or missing)
            res_sum_score = None
            if tool_eval["actual_called_tool"]:
                slot_cov = slot_eval["context_slot_accuracy"]
                ans_len = len(final_ans.split())
                if 40 <= ans_len <= 300:
                    conciseness = 1.0
                elif ans_len < 40:
                    conciseness = max(0.5, ans_len / 40.0)
                else:
                    conciseness = max(0.5, 300.0 / max(ans_len, 1))
                res_sum_score = round(min(1.0, max(0.0, 0.7 * slot_cov + 0.3 * conciseness)), 3)
            elif tool_eval.get("missing_tool_call"):
                # Phương án 2: Khi lượt yêu cầu gọi tool mà mô hình quên gọi (Missing Tool Call),
                # phạt điểm tóm tắt công cụ = 0.0 vì không có kết quả từ công cụ để tóm tắt
                res_sum_score = 0.0

            # - Follow-up understanding (for turn >= 2)
            follow_up_score = None
            if turn_num >= 2:
                slot_s = slot_eval["context_slot_accuracy"]
                coref_s = coref_eval["coreference_accuracy"] if coref_eval["has_coreference"] else 1.0
                # Check repetition with prior answer
                rep_penalty = 0.0
                if prior_answers:
                    last_ans = prior_answers[-1].lower()
                    overlap = sum(1 for w in set(final_ans.lower().split()) if w in last_ans)
                    ratio = overlap / max(len(set(final_ans.lower().split())), 1)
                    if ratio > 0.85:
                        rep_penalty = 0.3
                follow_up_score = round(max(0.0, 0.5 * slot_s + 0.3 * coref_s + 0.2 * (1.0 - rep_penalty)), 3)

            # - Final answer quality
            slot_part = slot_eval["context_slot_accuracy"]
            ground_val = claim_eval.get("grounded_claim_precision")
            ground_part = ground_val if ground_val is not None else slot_part
            structure_part = 1.0 if len(final_ans.split()) >= 25 else 0.5
            final_ans_quality = round(0.4 * slot_part + 0.4 * ground_part + 0.2 * structure_part, 3)

            cumulative_context += f"\nUser lượt {turn_num}: {user_msg}\nAssistant lượt {turn_num}: {final_ans}\n"
            prior_answers.append(final_ans)

            # 6. BinEval Checklist Evaluation (Aspect-Based Binary Scoring)
            bineval_eval = self.bineval.evaluate_turn(
                turn_data=turn_data,
                turn_gt=turn_gt,
                tool_eval=tool_eval,
                slot_eval=slot_eval,
                coref_eval=coref_eval,
                claim_eval=claim_eval,
                final_ans_quality=final_ans_quality
            )

            evaluated_turn = {
                "turn": turn_num,
                "user_message": user_msg,
                "tool_evaluation": tool_eval,
                "slot_evaluation": slot_eval,
                "coreference_evaluation": coref_eval,
                "claim_evaluation": claim_eval,
                "bineval_evaluation": bineval_eval,
                "result_summarization_score": res_sum_score,
                "follow_up_understanding": follow_up_score,
                "final_answer_quality": final_ans_quality
            }
            evaluated_turns.append(evaluated_turn)

        # 6. Conversation-Level Metrics Across 3 Pillars
        convo_metrics = self._aggregate_conversation_metrics(evaluated_turns, scenario_gt)

        return {
            "sample_id": sample_id,
            "turns": evaluated_turns,
            "conversation_metrics": convo_metrics
        }

    def _aggregate_conversation_metrics(
        self,
        evaluated_turns: List[Dict[str, Any]],
        scenario_gt: Dict[str, Any]
    ) -> Dict[str, Any]:
        total_turns = len(evaluated_turns)
        if total_turns == 0:
            return {"task_success": 0.0}

        # --- PILLAR A: TOOL USE ---
        tool_decisions = [t["tool_evaluation"]["tool_decision_accuracy"] for t in evaluated_turns]
        avg_tool_decision = sum(tool_decisions) / total_turns

        tool_arg_scores = [
            t["tool_evaluation"]["tool_argument_accuracy"]
            for t in evaluated_turns
            if t["tool_evaluation"]["actual_called_tool"] and t["tool_evaluation"]["tool_argument_accuracy"] is not None
        ]
        avg_tool_arg = sum(tool_arg_scores) / len(tool_arg_scores) if tool_arg_scores else 1.0

        # Tool switching accuracy across transitions (e.g. Turn 1 -> Turn 2, Turn 2 -> Turn 3)
        switch_matches = 0
        total_transitions = max(total_turns - 1, 1)
        for i in range(total_turns - 1):
            curr_actual = evaluated_turns[i]["tool_evaluation"]["actual_called_tool"]
            next_actual = evaluated_turns[i+1]["tool_evaluation"]["actual_called_tool"]
            curr_should = evaluated_turns[i]["tool_evaluation"]["should_call_tool"]
            next_should = evaluated_turns[i+1]["tool_evaluation"]["should_call_tool"]
            # Correct transition if both states matched expectations
            if (curr_actual == curr_should) and (next_actual == next_should):
                switch_matches += 1
        tool_switching_acc = switch_matches / total_transitions

        unnecessary_calls = sum(1 for t in evaluated_turns if t["tool_evaluation"]["unnecessary_tool_call"])
        missing_calls = sum(1 for t in evaluated_turns if t["tool_evaluation"]["missing_tool_call"])

        # --- PILLAR B: MULTI-TURN ---
        slot_scores = [
            t["slot_evaluation"]["context_slot_accuracy"]
            for t in evaluated_turns
            if t["slot_evaluation"]["total_required_slots"] > 0
        ]
        avg_slot_acc = (sum(slot_scores) / len(slot_scores)) if slot_scores else 1.0

        coref_scores = [
            t["coreference_evaluation"]["coreference_accuracy"]
            for t in evaluated_turns
            if t["coreference_evaluation"]["has_coreference"]
        ]
        avg_coref_acc = (sum(coref_scores) / len(coref_scores)) if coref_scores else 1.0

        follow_up_scores = [
            t["follow_up_understanding"]
            for t in evaluated_turns
            if t["follow_up_understanding"] is not None
        ]
        avg_follow_up = sum(follow_up_scores) / len(follow_up_scores) if follow_up_scores else 1.0

        # State tracking accuracy: consistency across state constraints
        contradiction_rates = [t["claim_evaluation"]["contradiction_rate"] for t in evaluated_turns]
        avg_contradiction_rate = sum(contradiction_rates) / total_turns if contradiction_rates else 0.0
        state_tracking_acc = max(0.0, avg_slot_acc * (1.0 - avg_contradiction_rate))
        cross_turn_consistency = round(1.0 - avg_contradiction_rate, 3)

        # --- PILLAR C: ANSWER QUALITY ---
        res_sum_scores = [
            t["result_summarization_score"]
            for t in evaluated_turns
            if t["result_summarization_score"] is not None
        ]
        avg_res_sum = sum(res_sum_scores) / len(res_sum_scores) if res_sum_scores else 1.0

        final_ans_qualities = [t["final_answer_quality"] for t in evaluated_turns]
        avg_final_quality = sum(final_ans_qualities) / total_turns

        claim_precisions = [
            t["claim_evaluation"]["grounded_claim_precision"]
            for t in evaluated_turns
            if t["claim_evaluation"].get("grounded_claim_precision") is not None
        ]
        avg_grounded_precision = (sum(claim_precisions) / len(claim_precisions)) if claim_precisions else 1.0

        unsupported_rates = [t["claim_evaluation"]["unsupported_claim_rate"] for t in evaluated_turns]
        avg_unsupported_rate = sum(unsupported_rates) / total_turns if unsupported_rates else 0.0

        # --- TASK SUCCESS ---
        task_success = (
            (avg_tool_decision == 1.0) and
            (missing_calls == 0) and
            (avg_slot_acc >= 0.6) and
            (avg_contradiction_rate == 0.0) and
            (avg_grounded_precision >= 0.75)
        )

        # --- BINEVAL CHECKLIST METRICS ---
        bineval_tool_scores = [t["bineval_evaluation"]["aspects"]["tool_proficiency"]["score"] for t in evaluated_turns]
        avg_bineval_tool = sum(bineval_tool_scores) / total_turns if bineval_tool_scores else 1.0

        bineval_context_scores = [t["bineval_evaluation"]["aspects"]["context_continuity"]["score"] for t in evaluated_turns]
        avg_bineval_context = sum(bineval_context_scores) / total_turns if bineval_context_scores else 1.0

        bineval_fact_scores = [t["bineval_evaluation"]["aspects"]["factuality_quality"]["score"] for t in evaluated_turns]
        avg_bineval_fact = sum(bineval_fact_scores) / total_turns if bineval_fact_scores else 1.0

        bineval_turn_scores = [t["bineval_evaluation"]["turn_score"] for t in evaluated_turns]
        avg_bineval_overall = sum(bineval_turn_scores) / total_turns if bineval_turn_scores else 1.0

        return {
            "task_success": 1.0 if task_success else 0.0,

            # A. Tool Use
            "tool_decision_accuracy": round(avg_tool_decision, 3),
            "tool_argument_accuracy": round(avg_tool_arg, 3),
            "tool_switching_accuracy": round(tool_switching_acc, 3),
            "missing_tool_call_rate": round(missing_calls / total_turns, 3),
            "unnecessary_tool_call_rate": round(unnecessary_calls / total_turns, 3),

            # B. Multi-Turn
            "context_slot_accuracy": round(avg_slot_acc, 3),
            "coreference_accuracy": round(avg_coref_acc, 3),
            "follow_up_understanding": round(avg_follow_up, 3),
            "state_tracking_accuracy": round(state_tracking_acc, 3),
            "cross_turn_consistency": cross_turn_consistency,

            # C. Answer Quality
            "result_summarization_score": round(avg_res_sum, 3),
            "final_answer_quality": round(avg_final_quality, 3),
            "grounded_claim_precision": round(avg_grounded_precision, 3),
            "unsupported_claim_rate": round(avg_unsupported_rate, 3),
            "contradiction_rate": round(avg_contradiction_rate, 3),

            # D. BinEval Checklist (Binary Scoring)
            "bineval_tool_score": round(avg_bineval_tool, 3),
            "bineval_context_score": round(avg_bineval_context, 3),
            "bineval_factuality_score": round(avg_bineval_fact, 3),
            "bineval_overall_score": round(avg_bineval_overall, 3)
        }


def normalize_trajectory_record(record: Dict[str, Any], gt_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a trajectory record to ensure compatibility with TrajectoryEvaluator:
    - Resolves sample_id (matches against gt_map or falls back from conversation_id)
    - Converts OpenAI / ChatML format (messages list) to structured turns list
    """
    norm_rec = dict(record)

    # 1. Resolve sample_id
    sample_id = norm_rec.get("sample_id")
    if not sample_id or sample_id not in gt_map:
        cid = norm_rec.get("conversation_id", "")
        first_user_msg = ""
        if "messages" in norm_rec:
            for m in norm_rec.get("messages", []):
                if m.get("role") == "user":
                    first_user_msg = m.get("content", "").strip()
                    break
        elif "turns" in norm_rec and norm_rec["turns"]:
            first_user_msg = norm_rec["turns"][0].get("user_message", "").strip()

        matched_sid = None
        if cid and cid in gt_map:
            matched_sid = cid

        if not matched_sid and first_user_msg:
            for sid, gt in gt_map.items():
                gt_turns = gt.get("scenario", {}).get("turns", [])
                gt_first = gt_turns[0].get("user_message", "").strip() if gt_turns else ""
                if gt_first and (gt_first == first_user_msg or gt_first in first_user_msg or first_user_msg in gt_first):
                    matched_sid = sid
                    break

        if not matched_sid and cid.startswith("conversation_"):
            try:
                num = int(cid.split("_")[1])
                target_sid = f"sample_{num + 13:03d}"
                if target_sid in gt_map:
                    matched_sid = target_sid
            except Exception:
                pass

        norm_rec["sample_id"] = matched_sid or sample_id or cid

    # 2. Convert 'messages' (ChatML / OpenAI format) -> 'turns'
    if ("turns" not in norm_rec or not norm_rec["turns"]) and "messages" in norm_rec:
        raw_msgs = norm_rec.get("messages", [])
        turns = []
        current_turn = None
        turn_num = 1

        for msg in raw_msgs:
            role = msg.get("role")
            content = msg.get("content") or ""

            if role == "user":
                if current_turn and current_turn.get("user_message"):
                    turns.append(current_turn)
                    turn_num += 1
                current_turn = {
                    "turn": turn_num,
                    "user_message": content.strip(),
                    "agent_response": {"type": "direct_answer"},
                    "tool_result": None,
                    "final_answer": ""
                }
            elif role == "assistant":
                if not current_turn:
                    continue
                tool_calls = msg.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    fn = tool_calls[0].get("function", {})
                    fn_name = fn.get("name", "web_search")
                    raw_args = fn.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except Exception:
                            raw_args = {"query": raw_args}
                    current_turn["agent_response"] = {
                        "type": "tool_call",
                        "tool_call": {
                            "name": fn_name,
                            "arguments": raw_args
                        }
                    }
                elif content:
                    current_turn["final_answer"] = content.strip()
            elif role == "tool":
                if current_turn:
                    raw_content = content.strip()
                    results = []
                    if raw_content.startswith("[") or raw_content.startswith("{"):
                        try:
                            parsed = json.loads(raw_content)
                            if isinstance(parsed, list):
                                results = parsed
                            elif isinstance(parsed, dict):
                                results = [parsed]
                        except Exception:
                            results = [{"snippet": raw_content}]
                    else:
                        results = [{"snippet": raw_content}]
                    current_turn["tool_result"] = {
                        "type": "tool_result",
                        "tool_name": current_turn.get("agent_response", {}).get("tool_call", {}).get("name", "web_search"),
                        "results": results
                    }
        if current_turn and current_turn.get("user_message"):
            turns.append(current_turn)

        norm_rec["turns"] = turns

    return norm_rec


# ==============================================================================
# 4. Batch Evaluator & Main Entrypoint
# ==============================================================================

async def evaluate_single_batch(
    batch_file: Path,
    output_dir: Path,
    evaluator: TrajectoryEvaluator,
    model_name: str,
    enrich_model: str = "gemini-3.5-flash-lite",
    auto_enrich: bool = True,
    force_enrich: bool = False
) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"📦 EVALUATING BATCH FILE: {batch_file.name}")
    print(f"{'='*75}")

    m = re.search(r"samples__(.+)\.json", batch_file.name)
    if m:
        out_filename = f"eval__{m.group(1)}.json"
    else:
        out_filename = f"eval__{batch_file.stem}.json"

    out_path = output_dir / out_filename

    # Step 1: Ground Truth handling (enrich with Gemini 3.5 Flash Lite if missing or force_enrich)
    gt_dir = WORKSPACE_ROOT / "eval/tool_call/datasets/multi_turn/ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / f"gt_{batch_file.name}"

    if auto_enrich and (not gt_path.exists() or force_enrich):
        if enrich_batch_dataset is None:
            print(f"❌ enrich_batch_dataset could not be imported.")
            return {}
        print(f"🔄 Enriching Ground Truth with {enrich_model} for {batch_file.name}...")
        scenarios, gt_path = await enrich_batch_dataset(
            input_file=str(batch_file),
            output_file=str(gt_path),
            model=enrich_model,
            force=force_enrich
        )
    elif not gt_path.exists():
        print(f"❌ Ground truth file not found at {gt_path.name} and auto-enrich is disabled.")
        return {}

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_scenarios = json.load(f)

    gt_map = {}
    for s in gt_scenarios:
        if s.get("sample_id"):
            gt_map[s["sample_id"]] = s
        if s.get("conversation_id"):
            gt_map[s["conversation_id"]] = s

    # Step 2: Load raw batch conversations
    with open(batch_file, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    if isinstance(raw_items, dict):
        raw_items = raw_items.get("conversations", raw_items.get("samples", [raw_items]))

    trajectories = [normalize_trajectory_record(t, gt_map) for t in raw_items]
    print(f"📋 Loaded {len(trajectories)} scenario trajectory/trajectories from batch")

    # Step 3: Evaluate scenarios
    eval_results = []
    for traj in trajectories:
        sid = traj.get("sample_id")
        scenario_gt = gt_map.get(sid, {})
        print(f"🔍 Evaluating scenario: {sid}...")
        res = await evaluator.evaluate_scenario(traj, scenario_gt)
        eval_results.append(res)
        cm = res["conversation_metrics"]
        print(f"  ✓ Task Success: {cm['task_success']} | Tool Acc: {cm['tool_decision_accuracy']*100:.0f}% | Slot Acc: {cm['context_slot_accuracy']*100:.0f}% | Grounded: {cm['grounded_claim_precision']*100:.1f}%")

    # Step 4: Aggregate metrics
    def avg_metric(key: str) -> float:
        vals = [r["conversation_metrics"][key] for r in eval_results if r.get("conversation_metrics", {}).get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary_metrics = {
        "task_success_rate": avg_metric("task_success"),

        # A. Tool Use
        "tool_decision_accuracy": avg_metric("tool_decision_accuracy"),
        "tool_argument_accuracy": avg_metric("tool_argument_accuracy"),
        "tool_switching_accuracy": avg_metric("tool_switching_accuracy"),
        "missing_tool_call_rate": avg_metric("missing_tool_call_rate"),
        "unnecessary_tool_call_rate": avg_metric("unnecessary_tool_call_rate"),

        # B. Multi-Turn
        "context_slot_accuracy": avg_metric("context_slot_accuracy"),
        "coreference_accuracy": avg_metric("coreference_accuracy"),
        "follow_up_understanding": avg_metric("follow_up_understanding"),
        "state_tracking_accuracy": avg_metric("state_tracking_accuracy"),
        "cross_turn_consistency": avg_metric("cross_turn_consistency"),

        # C. Answer Quality
        "result_summarization_score": avg_metric("result_summarization_score"),
        "final_answer_quality": avg_metric("final_answer_quality"),
        "grounded_claim_precision": avg_metric("grounded_claim_precision"),
        "unsupported_claim_rate": avg_metric("unsupported_claim_rate"),
        "contradiction_rate": avg_metric("contradiction_rate"),

        # D. BinEval Checklist (Binary Scoring)
        "bineval_tool_score": avg_metric("bineval_tool_score"),
        "bineval_context_score": avg_metric("bineval_context_score"),
        "bineval_factuality_score": avg_metric("bineval_factuality_score"),
        "bineval_overall_score": avg_metric("bineval_overall_score")
    }

    report = {
        "batch_file": batch_file.name,
        "model": model_name,
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_scenarios": len(eval_results),
        "summary_metrics": summary_metrics,
        "scenarios": eval_results
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*75}")
    print(f"🏁 BATCH EVALUATION COMPLETE FOR {batch_file.name}")
    print(f"🏆 Task Success Rate:              {summary_metrics['task_success_rate']*100:.1f}%")
    print(f"🎯 BinEval Overall Score:          {summary_metrics['bineval_overall_score']*100:.1f}%")
    print(f"📁 Evaluation report saved to:     {out_path.resolve()}")
    print(f"{'='*75}\n")
    return report


async def main_async():
    parser = argparse.ArgumentParser(description="Multi-Turn Trajectory Evaluator with 3-Pillar Taxonomy")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL", "gemma-4-31B-it"), help="Model name (or set MODEL in .env)")
    parser.add_argument("--trajectory", type=str, default=None, help="Path to agent_output.jsonl")
    parser.add_argument("--dataset", type=str, default="eval/tool_call/datasets/multi_turn/samples.json")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--batch-file", type=str, default=None, help="Path to single batch file (e.g. eval/tool_call/datasets/multi_turn/batches/samples__20260925_123650.json)")
    parser.add_argument("--batch-dir", type=str, default=None, help="Directory containing batch files (e.g. eval/tool_call/datasets/multi_turn/batches)")
    parser.add_argument("--output-dir", type=str, default="eval/tool_call/results/eval", help="Directory where batch evaluation results will be saved (default: eval/tool_call/results/eval)")
    parser.add_argument("--skip-llm-claim-eval", action="store_true", help="Skip LLM claim verification")
    parser.add_argument("--claim-model", type=str, default=os.environ.get("MODEL", "gemma-4-31B-it"), help="Model for atomic claim evaluation (default from .env: gemma-4-31B-it)")
    parser.add_argument("--claim-base-url", type=str, default=os.environ.get("BASE_URL", "https://more-starring-administration-cemetery.trycloudflare.com/v1"), help="Base URL for claim evaluation model")
    parser.add_argument("--enrich-model", type=str, default="gemini-3.5-flash-lite", help="Model to enrich Ground Truth (default: gemini-3.5-flash-lite)")
    parser.add_argument("--no-auto-enrich", action="store_true", help="Disable automatic Ground Truth enrichment if missing")
    parser.add_argument("--force-enrich", action="store_true", help="Force re-enrichment of Ground Truth even if cache exists")
    parser.add_argument("--sample-id", type=str, default=None, help="Evaluate only a specific sample_id (e.g. sample_013)")
    parser.add_argument("--from-sample", type=str, default=None, help="Evaluate starting from a specific sample_id onwards (e.g. sample_011)")
    parser.add_argument("--to-sample", type=str, default=None, help="Evaluate up to a specific sample_id (e.g. sample_013)")

    args = parser.parse_args()

    claim_verifier = AtomicClaimVerifier(
        base_url=args.claim_base_url,
        model=args.claim_model,
        enabled=not args.skip_llm_claim_eval
    )
    evaluator = TrajectoryEvaluator(claim_verifier=claim_verifier)

    # ── Branch 1: Batch Mode Evaluation ───────────────────────────────────────
    if args.batch_file or args.batch_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (WORKSPACE_ROOT / output_dir).resolve()

        if args.batch_file:
            b_path = Path(args.batch_file)
            if not b_path.is_absolute():
                b_path = (WORKSPACE_ROOT / b_path).resolve()
            if not b_path.exists():
                print(f"❌ Batch file not found: {b_path}")
                sys.exit(1)
            await evaluate_single_batch(
                batch_file=b_path,
                output_dir=output_dir,
                evaluator=evaluator,
                model_name=args.model,
                enrich_model=args.enrich_model,
                auto_enrich=not args.no_auto_enrich,
                force_enrich=args.force_enrich
            )
            return

        if args.batch_dir:
            b_dir = Path(args.batch_dir)
            if not b_dir.is_absolute():
                b_dir = (WORKSPACE_ROOT / b_dir).resolve()
            if not b_dir.exists():
                print(f"❌ Batch directory not found: {b_dir}")
                sys.exit(1)
            batch_files = sorted(list(b_dir.glob("samples__*.json")))
            if not batch_files:
                batch_files = sorted(list(b_dir.glob("*.json")))
            print(f"📂 Found {len(batch_files)} batch file(s) in {b_dir}")
            for bf in batch_files:
                await evaluate_single_batch(
                    batch_file=bf,
                    output_dir=output_dir,
                    evaluator=evaluator,
                    model_name=args.model,
                    enrich_model=args.enrich_model,
                    auto_enrich=not args.no_auto_enrich,
                    force_enrich=args.force_enrich
                )
            return

    # ── Branch 2: Standard Single-Trajectory Evaluation ────────────────────────
    if args.trajectory:
        traj_path = Path(args.trajectory)
    else:
        traj_path = WORKSPACE_ROOT / f"eval/tool_call/results/{args.model}/agent_output.jsonl"

    if not traj_path.exists():
        print(f"❌ Trajectory file not found: {traj_path}")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (WORKSPACE_ROOT / dataset_path).resolve()

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples_raw = json.load(f)
    gt_map = {s.get("sample_id"): s for s in samples_raw}

    raw_trajectories = []
    # Support both JSON Array file (.json) and JSON Lines (.jsonl)
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                raw_trajectories = data
            elif isinstance(data, dict):
                raw_trajectories = data.get("trajectories", [data])
    except Exception:
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    raw_trajectories.append(json.loads(line_str))

    trajectories = [normalize_trajectory_record(t, gt_map) for t in raw_trajectories]

    # Filter trajectories by sample_id, from_sample, to_sample
    if args.sample_id:
        matched = [t for t in trajectories if t.get("sample_id") == args.sample_id]
        if not matched:
            print(f"❌ Scenario {args.sample_id} not found in {traj_path}")
            sys.exit(1)
        trajectories = matched
        print(f"🎯 Filtered for single scenario: {args.sample_id}")
    elif args.from_sample or args.to_sample:
        from_idx = 0
        to_idx = len(trajectories)
        if args.from_sample:
            for i, t in enumerate(trajectories):
                if t.get("sample_id") == args.from_sample:
                    from_idx = i
                    break
        if args.to_sample:
            for i, t in enumerate(trajectories):
                if t.get("sample_id") == args.to_sample:
                    to_idx = i + 1
                    break
        trajectories = trajectories[from_idx:to_idx]
        print(f"🎯 Filtered range: {len(trajectories)} scenarios ({args.from_sample or 'start'} -> {args.to_sample or 'end'})")

    print(f"\n{'='*75}")
    print(f"🎯 Objective Multi-Turn Trajectory Evaluator | Model: {args.model}")
    print(f"📂 Trajectory source: {traj_path}")
    print(f"📋 Loaded {len(trajectories)} scenario trajectory/trajectories to evaluate")
    print(f"{'='*75}\n")

    eval_results = []
    for traj in trajectories:
        sid = traj.get("sample_id")
        print(f"🔍 Evaluating scenario: {sid}...")
        scenario_gt = gt_map.get(sid, {})
        res = await evaluator.evaluate_scenario(traj, scenario_gt)
        eval_results.append(res)
        cm = res["conversation_metrics"]
        print(f"  ✓ Task Success: {cm['task_success']} | Tool Acc: {cm['tool_decision_accuracy']*100:.0f}% | Slot Acc: {cm['context_slot_accuracy']*100:.0f}% | Grounded: {cm['grounded_claim_precision']*100:.1f}%")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = WORKSPACE_ROOT / f"eval/tool_call/results/{args.model}/evaluation_metrics.json"

    # Merge into existing evaluation_metrics.json if evaluating a subset
    final_scenarios_map = {}
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for sc in existing_data.get("scenarios", []):
                    if sc.get("sample_id"):
                        final_scenarios_map[sc["sample_id"]] = sc
        except Exception as e:
            print(f"⚠️ Could not read existing {out_path.name}: {e}")

    # Overwrite or insert newly evaluated scenarios
    for res in eval_results:
        final_scenarios_map[res["sample_id"]] = res

    # Sort scenarios by ID for neatness
    def sort_key(s_id: str):
        m = re.match(r"^sample_(\d+)$", s_id)
        return int(m.group(1)) if m else 999999

    merged_scenarios = sorted(final_scenarios_map.values(), key=lambda s: sort_key(s.get("sample_id", "")))
    total_scenarios = len(merged_scenarios)

    def avg_metric(key: str) -> float:
        vals = [r["conversation_metrics"][key] for r in merged_scenarios if r.get("conversation_metrics", {}).get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    summary_metrics = {
        "task_success_rate": avg_metric("task_success"),

        # A. Tool Use
        "tool_decision_accuracy": avg_metric("tool_decision_accuracy"),
        "tool_argument_accuracy": avg_metric("tool_argument_accuracy"),
        "tool_switching_accuracy": avg_metric("tool_switching_accuracy"),
        "missing_tool_call_rate": avg_metric("missing_tool_call_rate"),
        "unnecessary_tool_call_rate": avg_metric("unnecessary_tool_call_rate"),

        # B. Multi-Turn
        "context_slot_accuracy": avg_metric("context_slot_accuracy"),
        "coreference_accuracy": avg_metric("coreference_accuracy"),
        "follow_up_understanding": avg_metric("follow_up_understanding"),
        "state_tracking_accuracy": avg_metric("state_tracking_accuracy"),
        "cross_turn_consistency": avg_metric("cross_turn_consistency"),

        # C. Answer Quality
        "result_summarization_score": avg_metric("result_summarization_score"),
        "final_answer_quality": avg_metric("final_answer_quality"),
        "grounded_claim_precision": avg_metric("grounded_claim_precision"),
        "unsupported_claim_rate": avg_metric("unsupported_claim_rate"),
        "contradiction_rate": avg_metric("contradiction_rate"),

        # D. BinEval Checklist (Binary Scoring)
        "bineval_tool_score": avg_metric("bineval_tool_score"),
        "bineval_context_score": avg_metric("bineval_context_score"),
        "bineval_factuality_score": avg_metric("bineval_factuality_score"),
        "bineval_overall_score": avg_metric("bineval_overall_score")
    }

    summary = {
        "model": args.model,
        "evaluation_timestamp": datetime.now().isoformat(),
        "total_scenarios": total_scenarios,
        "summary_metrics": summary_metrics,
        "scenarios": merged_scenarios
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*75}")
    print(f"🏁 EVALUATION COMPLETE FOR {args.model}")
    print(f"🏆 Task Success Rate:              {summary_metrics['task_success_rate']*100:.1f}%\n")
    print(f"--- [A. TOOL USE] ---")
    print(f"  • tool_decision_accuracy:        {summary_metrics['tool_decision_accuracy']*100:.1f}%")
    print(f"  • tool_argument_accuracy:        {summary_metrics['tool_argument_accuracy']*100:.1f}%")
    print(f"  • tool_switching_accuracy:       {summary_metrics['tool_switching_accuracy']*100:.1f}%")
    print(f"  • missing_tool_call_rate:        {summary_metrics['missing_tool_call_rate']*100:.1f}%")
    print(f"  • unnecessary_tool_call_rate:    {summary_metrics['unnecessary_tool_call_rate']*100:.1f}%\n")
    print(f"--- [B. MULTI-TURN] ---")
    print(f"  • context_slot_accuracy:         {summary_metrics['context_slot_accuracy']*100:.1f}%")
    print(f"  • coreference_accuracy:          {summary_metrics['coreference_accuracy']*100:.1f}%")
    print(f"  • follow_up_understanding:       {summary_metrics['follow_up_understanding']*100:.1f}%")
    print(f"  • state_tracking_accuracy:       {summary_metrics['state_tracking_accuracy']*100:.1f}%")
    print(f"  • cross_turn_consistency:        {summary_metrics['cross_turn_consistency']*100:.1f}%\n")
    print(f"--- [C. ANSWER QUALITY] ---")
    print(f"  • result_summarization_score:    {summary_metrics['result_summarization_score']*100:.1f}%")
    print(f"  • final_answer_quality:          {summary_metrics['final_answer_quality']*100:.1f}%")
    print(f"  • grounded_claim_precision:      {summary_metrics['grounded_claim_precision']*100:.1f}%")
    print(f"  • unsupported_claim_rate:        {summary_metrics['unsupported_claim_rate']*100:.1f}%")
    print(f"  • contradiction_rate:            {summary_metrics['contradiction_rate']*100:.1f}%\n")
    print(f"--- [D. BINEVAL CHECKLIST SCORING] ---")
    print(f"  • bineval_tool_score:            {summary_metrics['bineval_tool_score']*100:.1f}%")
    print(f"  • bineval_context_score:         {summary_metrics['bineval_context_score']*100:.1f}%")
    print(f"  • bineval_factuality_score:      {summary_metrics['bineval_factuality_score']*100:.1f}%")
    print(f"  • bineval_overall_score:         {summary_metrics['bineval_overall_score']*100:.1f}%")
    print(f"\n📁 Full report saved to: {out_path.resolve()}")
    print(f"{'='*75}\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
