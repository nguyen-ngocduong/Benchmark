"""
Enrich multi_turn benchmark scenarios in samples.json with structured Ground Truth.

Includes:
- tool_call: should_call, expected_name, required_arg_keywords
- coreference: referent_phrase, expected_target
- required_slots: dict of slot_name -> list of acceptable synonyms/aliases
- golden_reference_claims: list of atomic verifiable facts from source article
"""

import json
from pathlib import Path

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


def main():
    dataset_path = Path("eval/tool_call/datasets/multi_turn/samples.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    updated_count = 0
    for sample in samples:
        sid = sample.get("sample_id")
        if sid in GROUND_TRUTH_DATA:
            gt_turns = GROUND_TRUTH_DATA[sid]
            turns = sample.get("scenario", {}).get("turns", [])
            for t_idx, turn in enumerate(turns):
                if t_idx < len(gt_turns):
                    turn["ground_truth"] = gt_turns[t_idx]
            updated_count += 1

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully enriched {updated_count}/{len(samples)} scenarios with Ground Truth in {dataset_path}")


if __name__ == "__main__":
    main()
