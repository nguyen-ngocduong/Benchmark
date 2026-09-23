#!/usr/bin/env bash
# ==============================================================================
# Pipeline Automation Runner for Benchmark Evaluation
# Supports:
#   - Task 1: Summarization Evaluation (run_objective_summarization_eval.py)
#   - Task 2: Multi-Turn Tool Calling Trajectory & Evaluation
#       + Step 2a: run_agent_trajectory.py -> agent_output.json(l)
#       + Step 2b: run_trajectory_eval.py   -> evaluation_metrics.json
# ==============================================================================

set -eo pipefail

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Workspace root
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

# Detect Python interpreter
if [ -f "$WORKSPACE_DIR/.venv/bin/python" ]; then
    PYTHON="$WORKSPACE_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

# Default configuration
MODELS=("gemma4:e2b" "qwen3.5:0.8b")
ACTION="all"
SKIP_LLM_CLAIM=false
SPECIFIC_MODEL=""
SAMPLE_ID=""
FROM_SAMPLE=""
TO_SAMPLE=""

print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "============================================================================"
    echo "  🚀 BENCHMARK PIPELINE AUTOMATION RUNNER"
    echo "  Workspace: $WORKSPACE_DIR"
    echo "  Python:    $($PYTHON --version 2>&1) ($PYTHON)"
    echo "============================================================================"
    echo -e "${NC}"
}

print_help() {
    echo -e "${BOLD}CÁCH SỬ DỤNG:${NC}"
    echo "  ./run.sh [ACTION] [OPTIONS]"
    echo ""
    echo -e "${BOLD}ACTIONS (Tác vụ):${NC}"
    echo "  all             Chạy toàn bộ Task 1 (Summarization) và Task 2 (Multi-turn) (Mặc định)"
    echo "  task1           Chạy Task 1: Document Summarization Evaluation"
    echo "  task2           Chạy Task 2: Multi-Turn Trajectory Generation & Evaluation"
    echo "  trajectory      Chỉ chạy sinh vết Agent (Step 2a: run_agent_trajectory.py)"
    echo "  eval            Chỉ chạy chấm điểm vết (Step 2b: run_trajectory_eval.py)"
    echo "  enrich          Làm giàu dữ liệu Ground Truth cho cả Task 1 và Task 2"
    echo "  enrich-task1    Làm giàu dữ liệu Ground Truth cho Task 1 (Summarization)"
    echo "  enrich-task2    Làm giàu dữ liệu Ground Truth cho Task 2 (Multi-turn)"
    echo "  enrich-mentor   Chuyển đổi & làm giàu dữ liệu multi_turn/samples_new.json của mentor"
    echo ""
    echo -e "${BOLD}OPTIONS (Tùy chọn):${NC}"
    echo "  -m, --model <name>     Chỉ chạy cho 1 model cụ thể (ví dụ: gemma4:e2b hoặc qwen3.5:0.8b)"
    echo "                         Nếu không truyền, mặc định sẽ chạy cả 2 models."
    echo "  --sample-id <ID>       Chỉ chạy cho 1 sample cụ thể (ví dụ: sample_011 hoặc sample_013)"
    echo "  --from-sample <ID>     Chạy nối tiếp từ 1 sample trở đi (ví dụ: sample_011)"
    echo "  --to-sample <ID>       Chạy đến 1 sample cụ thể (ví dụ: sample_013 hoặc sample_015)"
    echo "  --skip-llm-claim       Bỏ qua LLM Judge cho Atomic Claim (chạy fallback heuristic)"
    echo "  --use-llm-claim        Bật LLM Judge cho Atomic Claim (mặc định đã bật)"
    echo "  -h, --help             Hiển thị trợ giúp này"
    echo ""
    echo -e "${BOLD}VÍ DỤ:${NC}"
    echo "  ./run.sh                                  # Chạy toàn bộ cho cả gemma4 và qwen3.5"
    echo "  ./run.sh -m gemma4:e2b                    # Chạy toàn bộ chỉ cho gemma4:e2b"
    echo "  ./run.sh enrich-mentor                    # Làm giàu file mẫu của mentor"
    echo "  ./run.sh enrich-mentor --sample-id sample_013 # Làm giàu riêng sample_013"
    echo "  ./run.sh enrich-mentor --from-sample sample_013 --to-sample sample_015 # Làm giàu từ sample_013 đến sample_015"
    echo "  ./run.sh enrich --sample-id sample_011    # Làm giàu dữ liệu cho riêng sample_011"
    echo "  ./run.sh enrich-task2 --from-sample sample_011 # Làm giàu Task 2 từ sample_011 trở đi"
    echo "  ./run.sh trajectory -m gemma4:e2b --sample-id sample_011  # Chỉ sinh vết cho sample_011"
    echo "  ./run.sh eval -m gemma4:e2b               # Chạy chấm điểm tất cả các samples"
    echo "  ./run.sh eval -m gemma4:e2b --sample-id sample_013  # Chỉ chấm điểm riêng sample_013"
    echo "  ./run.sh eval -m gemma4:e2b --from-sample sample_011 --to-sample sample_013 # Chấm điểm từ sample_011 đến 013"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        all|task1|summarization|task2|multiturn|trajectory|eval|enrich|enrich-task1|enrich-task2|enrich-mentor)
            ACTION="$1"
            shift
            ;;
        -m|--model)
            SPECIFIC_MODEL="$2"
            shift 2
            ;;
        --sample-id)
            SAMPLE_ID="$2"
            shift 2
            ;;
        --from-sample)
            FROM_SAMPLE="$2"
            shift 2
            ;;
        --to-sample)
            TO_SAMPLE="$2"
            shift 2
            ;;
        --skip-llm-claim)
            SKIP_LLM_CLAIM=true
            shift
            ;;
        --use-llm-claim)
            SKIP_LLM_CLAIM=false
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Lỗi: Tùy chọn không hợp lệ '$1'${NC}"
            print_help
            exit 1
            ;;
    esac
done

# Map aliases
case "$ACTION" in
    summarization) ACTION="task1" ;;
    multiturn)     ACTION="task2" ;;
esac

# Check Ollama service
check_ollama() {
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️ CẢNH BÁO: Không kết nối được tới Ollama (http://localhost:11434).${NC}"
        echo -e "${YELLOW}   Vui lòng đảm bảo tiến trình 'ollama serve' đang chạy nếu sử dụng local models.${NC}\n"
    else
        echo -e "${GREEN}✓ Ollama service đang hoạt động trên http://localhost:11434${NC}"
    fi
}

run_task1_summarization() {
    local model="$1"
    echo -e "\n${BLUE}${BOLD}▶ [TASK 1] BẮT ĐẦU ĐÁNH GIÁ SUMMARIZATION: ${model}${NC}"
    echo -e "${CYAN}→ Script: run_objective_summarization_eval.py${NC}"
    
    $PYTHON -m eval.tool_call.scripts.run_objective_summarization_eval --model "$model"
    
    echo -e "${GREEN}✓ Hoàn tất Task 1 cho ${model}. Kết quả lưu tại: eval/tool_call/results/${model}/summarization_metrics.json${NC}"
}

run_step2a_trajectory() {
    local model="$1"
    echo -e "\n${BLUE}${BOLD}▶ [TASK 2 - Bước 2a] BẮT ĐẦU CHẠY AGENT TRAJECTORY: ${model}${NC}"
    echo -e "${CYAN}→ Script: run_agent_trajectory.py${NC}"
    
    local traj_args=(--model "$model")
    if [ -n "$SAMPLE_ID" ]; then
        traj_args+=(--sample-id "$SAMPLE_ID")
    fi
    if [ -n "$FROM_SAMPLE" ]; then
        traj_args+=(--from-sample "$FROM_SAMPLE")
    fi
    if [ -n "$TO_SAMPLE" ]; then
        traj_args+=(--to-sample "$TO_SAMPLE")
    fi
    
    $PYTHON -m eval.tool_call.scripts.run_agent_trajectory "${traj_args[@]}"
    
    echo -e "${GREEN}✓ Hoàn tất sinh vết cho ${model}. Log tương tác lưu tại: eval/tool_call/results/${model}/agent_output.json(l)${NC}"
}

run_step2b_evaluation() {
    local model="$1"
    echo -e "\n${BLUE}${BOLD}▶ [TASK 2 - Bước 2b] BẮT ĐẦU CHẤM ĐIỂM MULTI-TURN TRAJECTORY: ${model}${NC}"
    echo -e "${CYAN}→ Script: run_trajectory_eval.py${NC}"
    
    local eval_args=(--model "$model")
    if [ "$SKIP_LLM_CLAIM" = true ]; then
        eval_args+=(--skip-llm-claim-eval)
    fi
    if [ -n "$SAMPLE_ID" ]; then
        eval_args+=(--sample-id "$SAMPLE_ID")
    fi
    if [ -n "$FROM_SAMPLE" ]; then
        eval_args+=(--from-sample "$FROM_SAMPLE")
    fi
    if [ -n "$TO_SAMPLE" ]; then
        eval_args+=(--to-sample "$TO_SAMPLE")
    fi
    
    $PYTHON -m eval.tool_call.scripts.run_trajectory_eval "${eval_args[@]}"
    
    echo -e "${GREEN}✓ Hoàn tất chấm điểm cho ${model}. Báo cáo metric lưu tại: eval/tool_call/results/${model}/evaluation_metrics.json${NC}"
}

run_enrich_task1() {
    echo -e "\n${BLUE}${BOLD}▶ [LÀM GIÀU DỮ LIỆU - TASK 1] BẮT ĐẦU ENRICH SUMMARIZATION GROUND TRUTH${NC}"
    echo -e "${CYAN}→ Script: enrich_summarization_dataset.py${NC}"
    
    local enrich_args=()
    if [ -n "$SAMPLE_ID" ]; then
        enrich_args+=(--sample-id "$SAMPLE_ID")
    fi
    if [ -n "$FROM_SAMPLE" ]; then
        enrich_args+=(--from-sample "$FROM_SAMPLE")
    fi
    
    $PYTHON -m eval.tool_call.scripts.enrich_summarization_dataset "${enrich_args[@]}"
}

run_enrich_task2() {
    echo -e "\n${BLUE}${BOLD}▶ [LÀM GIÀU DỮ LIỆU - TASK 2] BẮT ĐẦU ENRICH MULTI-TURN GROUND TRUTH${NC}"
    echo -e "${CYAN}→ Script: enrich_dataset_ground_truth.py${NC}"
    
    local enrich_args=()
    if [ -n "$SAMPLE_ID" ]; then
        enrich_args+=(--sample-id "$SAMPLE_ID")
    fi
    if [ -n "$FROM_SAMPLE" ]; then
        enrich_args+=(--from-sample "$FROM_SAMPLE")
    fi
    
    $PYTHON -m eval.tool_call.scripts.enrich_dataset_ground_truth "${enrich_args[@]}"
}

run_enrich_mentor() {
    echo -e "\n${BLUE}${BOLD}▶ [LÀM GIÀU DỮ LIỆU - MENTOR MULTI-TURN] BẮT ĐẦU CHUYỂN ĐỔI VÀ ENRICH TỪ MULTI_TURN/SAMPLES_NEW.JSON${NC}"
    echo -e "${CYAN}→ Script: enrich_mentor_multiturn.py${NC}"
    
    local mentor_args=()
    if [ -n "$SAMPLE_ID" ]; then
        mentor_args+=(--sample-id "$SAMPLE_ID")
    fi
    if [ -n "$FROM_SAMPLE" ]; then
        mentor_args+=(--from-sample "$FROM_SAMPLE")
    fi
    if [ -n "$TO_SAMPLE" ]; then
        mentor_args+=(--to-sample "$TO_SAMPLE")
    fi
    
    $PYTHON -m eval.tool_call.scripts.enrich_mentor_multiturn "${mentor_args[@]}"
}

# Determine target models list
if [ -n "$SPECIFIC_MODEL" ]; then
    TARGET_MODELS=("$SPECIFIC_MODEL")
else
    TARGET_MODELS=("${MODELS[@]}")
fi

# Main execution logic
print_banner

# Dedicated branch for dataset enrichment (uses Gemini API, doesn't require Ollama)
if [[ "$ACTION" == "enrich" || "$ACTION" == "enrich-task1" || "$ACTION" == "enrich-task2" || "$ACTION" == "enrich-mentor" ]]; then
    START_TIME=$(date +%s)
    case "$ACTION" in
        enrich)
            run_enrich_task1
            run_enrich_task2
            ;;
        enrich-task1)
            run_enrich_task1
            ;;
        enrich-task2)
            run_enrich_task2
            ;;
        enrich-mentor)
            run_enrich_mentor
            ;;
    esac
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    echo -e "\n${GREEN}${BOLD}============================================================================${NC}"
    echo -e "${GREEN}${BOLD}🏁 HOÀN TẤT LÀM GIÀU DỮ LIỆU (Thời gian: ${TOTAL_TIME}s)!${NC}"
    echo -e "${GREEN}${BOLD}============================================================================${NC}"
    exit 0
fi

check_ollama

echo -e "${BOLD}Cấu hình thực thi:${NC}"
echo -e "  • Hành động (Action): ${YELLOW}$ACTION${NC}"
echo -e "  • Danh sách models:   ${YELLOW}${TARGET_MODELS[*]}${NC}"
echo -e "  • Skip LLM Claim API: ${YELLOW}$SKIP_LLM_CLAIM${NC}"
if [ -n "$SAMPLE_ID" ]; then
    echo -e "  • Target Sample:      ${YELLOW}$SAMPLE_ID${NC}"
fi
if [ -n "$FROM_SAMPLE" ]; then
    echo -e "  • Resuming From:      ${YELLOW}$FROM_SAMPLE${NC}"
fi
echo "----------------------------------------------------------------------------"

START_TIME=$(date +%s)

for model in "${TARGET_MODELS[@]}"; do
    echo -e "\n${BOLD}${CYAN}============================================================================${NC}"
    echo -e "${BOLD}${CYAN}🎯 TIẾN TRÌNH CHO MÔ HÌNH: ${model}${NC}"
    echo -e "${BOLD}${CYAN}============================================================================${NC}"

    case "$ACTION" in
        all)
            run_task1_summarization "$model"
            run_step2a_trajectory "$model"
            run_step2b_evaluation "$model"
            ;;
        task1)
            run_task1_summarization "$model"
            ;;
        task2)
            run_step2a_trajectory "$model"
            run_step2b_evaluation "$model"
            ;;
        trajectory)
            run_step2a_trajectory "$model"
            ;;
        eval)
            run_step2b_evaluation "$model"
            ;;
    esac
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo -e "\n${GREEN}${BOLD}============================================================================${NC}"
echo -e "${GREEN}${BOLD}🏁 HOÀN TẤT TOÀN BỘ TIẾN TRÌNH (Thời gian: ${TOTAL_TIME}s)!${NC}"
echo -e "${GREEN}${BOLD}============================================================================${NC}"
echo -e "Xem lại kết quả tại thư mục: ${CYAN}eval/tool_call/results/<model>/${NC}"
