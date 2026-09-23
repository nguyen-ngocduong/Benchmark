"""
Enrich multi_turn benchmark scenarios in samples.json with structured Ground Truth.

Includes:
- tool_call: should_call, expected_name, required_arg_keywords
- coreference: referent_phrase, expected_target
- required_slots: dict of slot_name -> list of acceptable synonyms/aliases
- golden_reference_claims: list of atomic verifiable facts from source article
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.tool_call.scripts.generate_multiturn_dataset import (
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_MODEL,
    generate_scenario_for_article
)

load_dotenv()

GROUND_TRUTH_DATA = {
    "sample_001": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["hermès", "khăn lụa", "tơ lụa", "nguyên liệu", "quy trình"]
            },
            "coreference": None,
            "required_slots": {
                "material_origin": ["brazil", "vườn ươm ở brazil", "vườn ươm"],
                "printing_workshop": ["lyon", "xưởng in lụa"]
            },
            "golden_reference_claims": [
                "Loại tơ dùng để dệt nên khăn lụa Hermès xuất xứ từ một vườn ươm ở Brazil.",
                "Mỗi mùa hơn mười phác thảo thiết kế mới được gửi đến xưởng in lụa của Hermès tại Lyon."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": {
                "referent_phrase": "nguồn gốc đó",
                "expected_target": ["brazil", "vườn ươm ở brazil", "vườn ươm"]
            },
            "required_slots": {
                "weaving_workshop": ["perrin", "xưởng dệt perrin", "pháp"],
                "duration": ["ba tháng", "gần ba tháng", "3 tháng"],
                "processes": ["quay lụa", "dệt", "khử keo"]
            },
            "golden_reference_claims": [
                "Những sợi kén tơ thô thu hoạch được mang đến xưởng dệt Perrin ở Pháp.",
                "Phải mất gần ba tháng lần lượt trải qua các công đoạn quay lụa, dệt và khử keo để tạo thành cuộn vải 100-150 mét."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "engraver_actions": ["giải mã", "sắc màu", "lớp phim", "tách màu", "bàn hắt sáng"],
                "drawing_tools": ["bút lông ngỗng", "bút chì", "bút điện"]
            },
            "golden_reference_claims": [
                "Thợ khắc giải mã các sắc màu và phác lại chúng trên nhiều lớp phim tương ứng.",
                "Sử dụng bàn hắt sáng trắng để tỉ mỉ tách từng màu và phác thảo bằng bút lông ngỗng, bút chì hay bút điện."
            ]
        }
    ],
    "sample_002": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["đá lạnh", "làm đẹp", "kết quả", "một tháng"]
            },
            "coreference": None,
            "required_slots": {
                "oil_skin": ["ít dầu", "đổ ít dầu", "mềm mịn", "dễ ăn phấn"],
                "pores": ["lỗ chân lông", "thu nhỏ"],
                "swelling": ["giảm sưng", "mặt gọn hơn", "bọng mắt"],
                "alertness": ["tỉnh táo", "ngái ngủ", "loại bỏ cảm giác ngái ngủ"]
            },
            "golden_reference_claims": [
                "Sau một tháng sử dụng đá lạnh, da đổ ít dầu, mềm mịn và dễ ăn phấn hơn.",
                "Lỗ chân lông được thu nhỏ đáng kể.",
                "Mắt và khuôn mặt giảm sưng nhiều, mặt gọn hơn và tỉnh táo trước khi đi làm."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": {
                "referent_phrase": "nguyên liệu chính vừa đề cập",
                "expected_target": ["đá lạnh", "dưa chuột"]
            },
            "required_slots": {
                "alternative_material": ["trà xanh", "bột trà xanh", "trà xanh nhật bản"],
                "preferred_material": ["dưa chuột", "nguyên liệu dưa chuột"]
            },
            "golden_reference_claims": [
                "Tác giả thỉnh thoảng đổi vị bằng cách dùng bột trà xanh Nhật Bản hòa với nước.",
                "Tuy nhiên nguyên liệu dưa chuột vẫn làm tác giả ưng ý nhất."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "skin_summary": ["kiềm dầu", "đổ ít dầu", "lỗ chân lông", "thu nhỏ", "giảm sưng"]
            },
            "golden_reference_claims": [
                "Phương pháp mang lại lợi ích giảm tiết dầu, se khít lỗ chân lông và giảm sưng bọng mắt."
            ]
        }
    ],
    "sample_003": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["chủ tịch quốc hội", "viện ksnd", "yêu cầu", "trọng tâm"]
            },
            "coreference": None,
            "required_slots": {
                "core_mandate": ["quyền công tố", "kiểm sát hoạt động tư pháp", "oan, sai", "oan sai"],
                "economic_civil": ["không hình sự hóa", "quan hệ dân sự", "kinh tế"]
            },
            "golden_reference_claims": [
                "Nâng cao chất lượng thực hành quyền công tố, kiểm sát hoạt động tư pháp.",
                "Không để xảy ra oan sai trong tố tụng hình sự, không hình sự hóa quan hệ dân sự, kinh tế."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "prevent_error": ["không để xảy ra oan, sai", "không oan sai", "không hình sự hóa"],
                "corruption_cases": ["tham nhũng", "kinh tế", "chức vụ", "ban chỉ đạo"]
            },
            "golden_reference_claims": [
                "Không để xảy ra oan sai trong tố tụng hình sự và không hình sự hóa quan hệ kinh tế, dân sự.",
                "Tập trung giải quyết tốt các vụ án kinh tế, tham nhũng, chức vụ thuộc diện Ban Chỉ đạo T.Ư theo dõi."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "apparatus": ["tinh gọn", "hiệu lực", "hiệu quả", "biên chế"],
                "discipline": ["kỷ luật", "kỷ cương", "thực thi công vụ"]
            },
            "golden_reference_claims": [
                "Sắp xếp cơ cấu tổ chức bộ máy tinh gọn, hiệu lực, hiệu quả và sử dụng hợp lý biên chế cán bộ.",
                "Giữ vững kỷ luật, kỷ cương trong thực thi công vụ."
            ]
        }
    ],
    "sample_004": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["bí thư", "nguyễn văn quảng", "đà nẵng", "lấy phiếu tín nhiệm"]
            },
            "coreference": None,
            "required_slots": {
                "speaker": ["nguyễn văn quảng", "bí thư thành ủy"],
                "session": ["kỳ họp thứ 15", "hđnd tp.đà nẵng", "12.12"]
            },
            "golden_reference_claims": [
                "Ngày 12.12 tại kỳ họp thứ 15 HĐND TP.Đà Nẵng, Bí thư Thành ủy Nguyễn Văn Quảng phát biểu về việc lấy phiếu tín nhiệm."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": {
                "referent_phrase": "nhân vật trên",
                "expected_target": ["nguyễn văn quảng", "bí thư thành ủy đà nẵng"]
            },
            "required_slots": {
                "self_improve": ["tự soi", "tự sửa", "tín nhiệm", "rèn luyện", "phấn đấu"],
                "state_efficiency": ["hiệu lực", "hiệu quả", "uy tín"]
            },
            "golden_reference_claims": [
                "Người được lấy phiếu tín nhiệm sẽ tự soi, tự sửa, thấy được mức độ tín nhiệm để tiếp tục rèn luyện, nâng cao hiệu quả công tác.",
                "Hoạt động giám sát nhằm nâng cao hiệu lực, hiệu quả của bộ máy nhà nước và đánh giá uy tín cán bộ."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "tackle_issues": ["né tránh", "đùn đẩy", "đùn đẩy trách nhiệm", "không làm đúng chức trách"]
            },
            "golden_reference_claims": [
                "Khắc phục, đẩy lùi tình trạng né tránh, đùn đẩy trách nhiệm, không làm đúng chức trách, quyền hạn của cán bộ, công chức."
            ]
        }
    ],
    "sample_005": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["thu hồi đất", "quốc hội", "dự thảo luật", "đại biểu"]
            },
            "coreference": None,
            "required_slots": {
                "speakers": ["mai văn hải", "hoàng văn cường"]
            },
            "golden_reference_claims": [
                "Các đại biểu Quốc hội Mai Văn Hải và Hoàng Văn Cường thảo luận về quy định thu hồi đất trong dự thảo luật."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "state_recovery": ["hạ tầng kỹ thuật", "hạ tầng xã hội", "công trình phúc lợi"],
                "negotiation_part": ["xây dựng nhà ở", "thỏa thuận", "tự thỏa thuận", "người dân và dn"]
            },
            "golden_reference_claims": [
                "ĐB Mai Văn Hải đề xuất Nhà nước chỉ thu hồi đất làm hạ tầng kỹ thuật, hạ tầng xã hội, phúc lợi công cộng.",
                "Phần diện tích xây dựng nhà ở thì phải thực hiện cơ chế tự thỏa thuận giữa người dân và doanh nghiệp theo Nghị quyết 18."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "legal_gap": ["khoảng trống", "không được đấu thầu", "không được thỏa thuận"],
                "proposal": ["khoản 27 điều 79", "đấu giá", "đấu thầu"]
            },
            "golden_reference_claims": [
                "ĐB Hoàng Văn Cường băn khoăn dự án không thỏa mãn tiêu chí sẽ tạo ra khoảng trống pháp luật: không đấu thầu, không thỏa thuận, không giao trực tiếp.",
                "Ông đề nghị sửa đổi khoản 27 điều 79 để tháo gỡ vướng mắc."
            ]
        }
    ],
    "sample_006": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["céline", "celine", "hedi slimane", "đổi tên"]
            },
            "coreference": None,
            "required_slots": {
                "creative_director": ["hedi slimane"],
                "reason": ["cân đối", "hiện đại hơn", "hiện đại"]
            },
            "golden_reference_claims": [
                "Thương hiệu Céline đổi thành Celine dưới sự chỉ đạo của Giám đốc Sáng tạo Hedi Slimane để cân đối và hiện đại hơn."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": {
                "referent_phrase": "thương hiệu trên",
                "expected_target": ["céline", "celine"]
            },
            "required_slots": {
                "balmain_move": ["balmain", "olivier rousteing", "logo mới", "73 năm"],
                "calvin_klein_move": ["calvin klein", "raf simons", "rời khỏi", "22 tháng 12"]
            },
            "golden_reference_claims": [
                "NTK Olivier Rousteing giới thiệu logo mới cho Balmain sau 73 năm.",
                "Ngày 22 tháng 12, Calvin Klein thông báo Raf Simons chính thức rời vị trí Giám đốc Sáng tạo."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": {
                "referent_phrase": "nhân vật vừa được nhắc đến",
                "expected_target": ["raf simons"]
            },
            "required_slots": {
                "product_line": ["calvin klein 205w39nyc", "205w39nyc"],
                "line_impact": ["tương lai", "thiếu vắng người cha đẻ", "tiếc nuối"]
            },
            "golden_reference_claims": [
                "Sự ra đi của Raf Simons đặt ra câu hỏi lớn về tương lai dòng sản phẩm Calvin Klein 205W39NYC khi thiếu vắng người cha đẻ."
            ]
        }
    ],
    "sample_007": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["thiều bảo trâm", "equal", "spotify", "chiến dịch"]
            },
            "coreference": None,
            "required_slots": {
                "times_square": ["quảng trường thời đại", "new york", "equal global"]
            },
            "golden_reference_claims": [
                "Thiều Bảo Trâm xuất hiện trong bộ ảnh chiến dịch EQUAL Global của Spotify được trình chiếu tại quảng trường Thời đại ở New York."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": {
                "referent_phrase": "chiến dịch đó",
                "expected_target": ["equal", "equal global", "spotify"]
            },
            "required_slots": {
                "album": ["after you", "mini album after you"],
                "release_time": ["7/6", "tháng 6"]
            },
            "golden_reference_claims": [
                "Bộ ảnh gắn liền với việc phát hành mini album 'after YOU' vào ngày 7/6 gồm năm ca khúc ballad."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "chart_top": ["số 1", "itunes việt nam", "bxh"],
                "korea_tour": ["hàn quốc", "simply k-pop", "13/6"]
            },
            "golden_reference_claims": [
                "Mini album 'after YOU' chiếm lĩnh vị trí số 1 BXH Album iTunes Việt Nam.",
                "Ngày 13/6, Thiều Bảo Trâm biểu diễn tại Hàn Quốc trong chương trình Simply K-Pop Con-tour."
            ]
        }
    ],
    "sample_008": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["pierpaolo piccioli", "valentino", "rời khỏi"]
            },
            "coreference": None,
            "required_slots": {
                "designer": ["pierpaolo piccioli"],
                "tenure": ["25 năm", "một phần tư thế kỷ", "giám đốc sáng tạo"]
            },
            "golden_reference_claims": [
                "Pierpaolo Piccioli tuyên bố rút lui khỏi vị trí giám đốc sáng tạo của Valentino sau 25 năm gắn bó."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "quote_essence": ["tiếng ý", "không phải mọi câu chuyện đều có khởi đầu và kết thúc", "bóng tối không thể nào chen chân"]
            },
            "golden_reference_claims": [
                "Trên Instagram bằng tiếng Ý, ông viết: 'Không phải mọi câu chuyện đều có khởi đầu và kết thúc, một số tồn tại mãi mãi, rực rỡ và mãnh liệt đến mức bóng tối không thể nào chen chân'."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": {
                "referent_phrase": "nhân vật trên",
                "expected_target": ["pierpaolo piccioli", "pierpaolo"]
            },
            "required_slots": {
                "ceo_jacopo": ["jacopo venturini", "tầm nhìn", "sáng tạo"],
                "chairman_rachid": ["rachid mohamed rachid", "chương rực rỡ"]
            },
            "golden_reference_claims": [
                "CEO Jacopo Venturini bày tỏ lòng biết ơn sâu sắc đến tầm nhìn và sự sáng tạo của Pierpaolo.",
                "Chủ tịch Rachid Mohamed Rachid cảm ơn ông đã viết nên một chương rực rỡ trong lịch sử Valentino."
            ]
        }
    ],
    "sample_009": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["lối sống", "cha mẹ", "định sẵn", "đơn điệu", "hiện đại"]
            },
            "coreference": None,
            "required_slots": {
                "routine": ["thức dậy", "đi làm", "trở về nhà", "đơn điệu", "sắp đặt"]
            },
            "golden_reference_claims": [
                "Nhiều người sống theo cách cha mẹ định sẵn, thói quen lặp lại đơn điệu, thiếu màu sắc và ước mơ."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "plywood_metaphor": ["tấm ván ép", "những tấm ván ép", "phẳng phiu", "không có một chút nứt vỡ"]
            },
            "golden_reference_claims": [
                "Tác giả ví những con người này sống như 'những tấm ván ép, phẳng phiu, không có một chút nứt vỡ nào, không có ước mơ hay tham vọng'."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "turning_point": ["một trong hai người", "hôn nhân", "bình lặng", "đơn điệu"],
                "decision": ["phá vỡ", "tìm kiếm", "khám phá bản thân", "trải nghiệm"]
            },
            "golden_reference_claims": [
                "Một trong hai người trong cuộc hôn nhân nhận ra sự đơn điệu và quyết định phá vỡ bầu không khí tĩnh lặng.",
                "Họ bắt đầu hành trình khám phá bản thân, tìm kiếm cảm xúc chân thật và sống động hơn."
            ]
        }
    ],
    "sample_010": [
        {
            "turn": 1,
            "tool_call": {
                "should_call": True,
                "expected_name": "web_search",
                "required_arg_keywords": ["bí mật tự nhiên", "mùa hè", "chăm sóc da", "thiên nhiên"]
            },
            "coreference": None,
            "required_slots": {
                "natural_elements": ["sữa tươi", "mật ong", "salad", "trà xanh", "vang đỏ", "nước chanh"]
            },
            "golden_reference_claims": [
                "Bài viết giới thiệu các bí mật tự nhiên chăm sóc da mùa hè: sữa tươi, mật ong, salad xanh, túi trà xanh, vang đỏ và nước chanh."
            ]
        },
        {
            "turn": 2,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": None,
            "required_slots": {
                "milk_lactic": ["axit lactic", "mềm mại", "mịn màng", "ngâm tay chân"],
                "honey_benefits": ["kháng khuẩn", "dưỡng ẩm", "bảo vệ da", "giữ ẩm"]
            },
            "golden_reference_claims": [
                "Sữa tươi chứa axit lactic giúp da mềm mại, mịn màng.",
                "Mật ong có tính kháng khuẩn và dưỡng ẩm tuyệt vời, phối hợp hoàn hảo cùng sữa tươi bảo vệ da."
            ]
        },
        {
            "turn": 3,
            "tool_call": {
                "should_call": False,
                "expected_name": None,
                "required_arg_keywords": []
            },
            "coreference": {
                "referent_phrase": "hai nguyên liệu trên",
                "expected_target": ["sữa tươi và mật ong", "sữa tươi", "mật ong"]
            },
            "required_slots": {
                "red_wine": ["vang đỏ", "polyphenol", "chống oxy hóa", "làm mềm da", "ngâm tay chân", "bồn tắm"],
                "lemon": ["nước chanh", "mái tóc", "chắc khỏe", "tươi sáng rạng rỡ"]
            },
            "golden_reference_claims": [
                "Vang đỏ chứa chất chống oxy hóa và polyphenol giúp làm mềm da, có thể ngâm chân tay hoặc thêm vào bồn tắm.",
                "Nước chanh giúp mái tóc chắc khỏe và mang lại vẻ tươi sáng rạng rỡ."
            ]
        }
    ]
}


async def enrich_multiturn_dataset(
    input_file: str = "eval/tool_call/datasets/summarization/samples_new.json",
    output_file: str = "eval/tool_call/datasets/multi_turn/samples.json",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = 0.2,
    sample_id: Optional[str] = None,
    from_sample: Optional[str] = None,
    single_test: bool = False,
    force: bool = False
):
    """
    Enriches multi-turn scenarios:
    1. Updates any existing scenarios (sample_001 -> sample_010) with curated GROUND_TRUTH_DATA.
    2. Reads newly loaded articles from samples_new.json (sample_011+).
    3. Calls LLM to generate 3-turn multi-turn scenarios with embedded ground truth.
    4. Appends them incrementally to multi_turn/samples.json.
    Supports single sample execution (--sample-id) and resumption from a sample (--from-sample).
    """
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load existing scenarios from multi_turn/samples.json
    existing_scenarios: List[Dict[str, Any]] = []
    existing_ids = set()

    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_scenarios = json.loads(content)
                    existing_ids = {s.get("sample_id") for s in existing_scenarios if s.get("sample_id")}
            print(f"📖 Loaded {len(existing_scenarios)} existing multi-turn scenarios from {output_path.name}")
        except Exception as e:
            print(f"⚠️ Could not load existing {output_path}: {e}")
            existing_scenarios = []
            existing_ids = set()

    # Apply curated GROUND_TRUTH_DATA to any existing scenarios that need it
    curated_updated = 0
    for sample in existing_scenarios:
        sid = sample.get("sample_id")
        if sid in GROUND_TRUTH_DATA:
            gt_turns = GROUND_TRUTH_DATA[sid]
            turns = sample.get("scenario", {}).get("turns", [])
            for t_idx, turn in enumerate(turns):
                if t_idx < len(gt_turns):
                    turn["ground_truth"] = gt_turns[t_idx]
            curated_updated += 1

    if curated_updated > 0:
        print(f"✓ Synced curated Ground Truth for {curated_updated} existing base scenarios.")

    # 2. Check input candidate articles (samples_new.json)
    if not input_path.exists():
        print(f"ℹ️ Input file {input_path} does not exist. Saving existing synced scenarios only.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_scenarios, f, ensure_ascii=False, indent=2)
        return

    with open(input_path, "r", encoding="utf-8") as f:
        new_samples = json.load(f)

    # Filter candidate samples by sample_id, from_sample, or single_test
    if sample_id:
        matched = [s for s in new_samples if (s.get("id") or s.get("sample_id")) == sample_id]
        if matched:
            new_samples = matched
            print(f"🎯 Filtered for single sample: {sample_id} ({len(new_samples)} matched in {input_path.name})")
        else:
            existing_match = next((s for s in existing_scenarios if s.get("sample_id") == sample_id), None)
            if existing_match:
                new_samples = [{
                    "id": sample_id,
                    "article": existing_match.get("source", {}).get("text", ""),
                    "label": existing_match.get("source", {}).get("label", 0)
                }]
                print(f"🎯 Filtered for existing scenario in dataset: {sample_id}")
            else:
                print(f"❌ Sample {sample_id} not found in {input_path.name} or {output_path.name}")
                return
    elif from_sample:
        from_idx = None
        for i, s in enumerate(new_samples):
            if (s.get("id") or s.get("sample_id")) == from_sample:
                from_idx = i
                break
        if from_idx is not None:
            new_samples = new_samples[from_idx:]
            print(f"🎯 Filtered from sample {from_sample} onwards ({len(new_samples)} samples in {input_path.name})")
        else:
            print(f"⚠️ Sample {from_sample} not found in {input_path.name}. Running all candidate samples.")

    if single_test and not sample_id:
        new_samples = new_samples[:1]
        print(f"🔍 Running in single-test mode: 1 sample ({new_samples[0].get('id') or new_samples[0].get('sample_id')})")
    elif not sample_id and not from_sample and not single_test:
        print(f"📋 Loaded {len(new_samples)} candidate samples from {input_path.name}")

    print(f"🌐 LLM Endpoint: {base_url} | Model: {model}\n")

    results = list(existing_scenarios)
    newly_added_count = 0

    for idx, sample in enumerate(new_samples, start=1):
        s_id = sample.get("id") or sample.get("sample_id") or f"sample_{idx:03d}"

        if s_id in existing_ids and not force and not sample_id:
            print(f"[{idx}/{len(new_samples)}] ⏭️ Skipping {s_id} (already exists in {output_path.name}).")
            continue

        print(f"[{idx}/{len(new_samples)}] ⏳ Generating Multi-Turn Ground Truth for {s_id}...")

        # If present in curated dictionary, use it
        if s_id in GROUND_TRUTH_DATA:
            print(f"    ✓ Found in curated GROUND_TRUTH_DATA.")
            article_text = sample.get("article") or sample.get("text", "")
            gt_turns = GROUND_TRUTH_DATA[s_id]
            scenario = {
                "sample_id": s_id,
                "source": {
                    "text": article_text,
                    "label": sample.get("label", 0)
                },
                "scenario": {
                    "objective": "multi_turn_context_tool_use",
                    "turns": [
                        {
                            "turn": t.get("turn", i + 1),
                            "user_message": f"Hãy tìm kiếm và tóm tắt thông tin về {s_id}",
                            "expected_behavior": {
                                "should_call_tool": t.get("tool_call", {}).get("should_call", False),
                                "expected_tool": t.get("tool_call", {}).get("expected_name"),
                                "expected_category": "news" if t.get("tool_call", {}).get("should_call") else None,
                                "query_intent": None,
                                "context_required": i > 0,
                                "state_required": i > 0
                            },
                            "ground_truth": t
                        }
                        for i, t in enumerate(gt_turns)
                    ]
                }
            }
        else:
            # Generate dynamically with LLM
            scenario = await generate_scenario_for_article(
                sample=sample,
                base_url=base_url,
                model=model,
                api_key=api_key,
                temperature=temperature
            )

        if scenario:
            existing_match_idx = next((i for i, x in enumerate(results) if x.get("sample_id") == s_id), None)
            if existing_match_idx is not None:
                results[existing_match_idx] = scenario
            else:
                results.append(scenario)
                existing_ids.add(s_id)

            newly_added_count += 1

            # Incremental save
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"    💾 Saved progress: {len(results)} total scenarios in {output_path.name}")
        else:
            print(f"    ❌ Failed to generate multi-turn scenario for {s_id}")

    print(f"\n{'='*70}")
    print(f"🎉 TASK 2 (MULTI-TURN) ENRICHMENT COMPLETE!")
    print(f"   • Newly added/updated: {newly_added_count} scenarios")
    print(f"   • Total scenarios now in {output_path.name}: {len(results)}")
    print(f"   • Output location: {output_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich multi-turn benchmark scenarios with Ground Truth (curated or via LLM from samples_new.json)"
    )
    parser.add_argument(
        "--input",
        default="eval/tool_call/datasets/summarization/samples_new.json",
        help="Path to input samples_new.json"
    )
    parser.add_argument(
        "--output",
        default="eval/tool_call/datasets/multi_turn/samples.json",
        help="Path to output multi_turn samples.json"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"LLM API Base URL (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="LLM API Key (default: from env GEMINI_API_KEY / LLM_API_KEY)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Enrich only a specific sample_id (e.g. sample_011)"
    )
    parser.add_argument(
        "--from-sample",
        type=str,
        default=None,
        help="Enrich starting from a specific sample_id onwards (e.g. sample_011)"
    )
    parser.add_argument(
        "--test-single",
        action="store_true",
        help="Enrich only 1 sample for verification"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force regenerate/overwrite existing scenarios with same ID"
    )

    args = parser.parse_args()

    asyncio.run(enrich_multiturn_dataset(
        input_file=args.input,
        output_file=args.output,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        sample_id=args.sample_id,
        from_sample=args.from_sample,
        single_test=args.test_single,
        force=args.force
    ))


if __name__ == "__main__":
    main()
