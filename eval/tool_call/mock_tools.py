"""
Mock tool responses for testing tool-calling without real API calls
"""

MOCK_RESPONSES = {
    "get_gold_price": {
        "current": {
            "status": "success",
            "data": {
                "SJC": {"buy": 89200000, "sell": 89800000, "unit": "VND/lượng"},
                "BTMC": {"buy": 89100000, "sell": 89700000, "unit": "VND/lượng"},
                "DOJI": {"buy": 89250000, "sell": 89900000, "unit": "VND/lượng"},
                "BTMH": {"buy": 89150000, "sell": 89750000, "unit": "VND/lượng"},
                "PNJ": {"buy": 89200000, "sell": 89850000, "unit": "VND/lượng"}
            },
            "timestamp": "2026-09-15T10:00:00+07:00",
            "note": "Giá vàng SJC hôm nay ổn định, tăng nhẹ 50,000 VND/lượng so với hôm qua."
        },
        "history": {
            "status": "success",
            "data": [
                {"date": "2026-09-14", "SJC_buy": 89150000, "SJC_sell": 89750000},
                {"date": "2026-09-13", "SJC_buy": 89100000, "SJC_sell": 89700000},
                {"date": "2026-09-12", "SJC_buy": 89050000, "SJC_sell": 89650000},
            ],
            "note": "Giá vàng SJC dao động trong khoảng 89-90 triệu VND/lượng trong 3 ngày gần đây."
        }
    },
    "get_weather": {
        "Hà Nội": {
            "status": "success",
            "current": {"temp": 32, "humidity": 75, "condition": "nắng", "wind": "15 km/h"},
            "forecast": [
                {"date": "2026-09-15", "temp_high": 34, "temp_low": 26, "condition": "nắng"},
                {"date": "2026-09-16", "temp_high": 33, "temp_low": 25, "condition": "mưa rào"}
            ]
        },
        "TP Hồ Chí Minh": {
            "status": "success",
            "current": {"temp": 34, "humidity": 80, "condition": "nắng nóng", "wind": "10 km/h"},
            "forecast": [
                {"date": "2026-09-15", "temp_high": 35, "temp_low": 27, "condition": "nắng nóng"},
                {"date": "2026-09-16", "temp_high": 34, "temp_low": 26, "condition": "mưa rào chiều"}
            ]
        },
        "Đà Nẵng": {
            "status": "success",
            "current": {"temp": 31, "humidity": 70, "condition": "nắng nhẹ", "wind": "20 km/h"},
            "forecast": [
                {"date": "2026-09-15", "temp_high": 32, "temp_low": 26, "condition": "nắng nhẹ"},
                {"date": "2026-09-16", "temp_high": 31, "temp_low": 25, "condition": "mây rải rác"}
            ]
        }
    },
    "execute_robot_command": {
        "set_light_on": {
            "status": "success",
            "action": "set_light",
            "location_id": "living_room",
            "light_state": "on",
            "message": "Đã bật đèn phòng khách."
        },
        "set_light_off": {
            "status": "success",
            "action": "set_light",
            "location_id": "living_room",
            "light_state": "off",
            "message": "Đã tắt đèn phòng khách."
        },
        "move": {
            "status": "success",
            "action": "move",
            "location_id": "kitchen",
            "message": "Di chuyển đến bếp thành công."
        },
        "inspect": {
            "status": "success",
            "action": "inspect",
            "location_id": "living_room",
            "message": "Không phát hiện bất thường trong phòng khách.",
            "details": {"light": "on", "temperature": 26, "humidity": 65}
        },
        "call": {
            "status": "success",
            "action": "call",
            "contact_id": "family_mom",
            "message": "Đã gọi cho mẹ."
        }
    },
    "search_web": {
        "default": {
            "status": "success",
            "results": [
                {"title": "Tin tức mới nhất", "snippet": "Thông tin cập nhật...", "url": "https://example.com/1"},
                {"title": "Báo cáo thị trường", "snippet": "Phân tích chi tiết...", "url": "https://example.com/2"}
            ]
        }
    },
    "query_knowledge_base": {
        "default": {
            "status": "success",
            "results": [
                {"chunk_id": "doc_001", "content": "Robot Dyno là robot phụ tá gia đình...", "score": 0.95},
                {"chunk_id": "doc_002", "content": "VinDynamics cung cấp giải pháp AI...", "score": 0.87}
            ]
        }
    }
}


def get_mock_response(tool_name: str, arguments: dict) -> dict:
    """
    Get mock response for a tool call

    Args:
        tool_name: Name of the tool
        arguments: Tool arguments

    Returns:
        Mock response dict
    """
    if tool_name == "get_gold_price":
        action = arguments.get("action", "current")
        if action == "history":
            return MOCK_RESPONSES["get_gold_price"]["history"]
        return MOCK_RESPONSES["get_gold_price"]["current"]

    elif tool_name == "get_weather":
        location = arguments.get("keyword", "")
        return MOCK_RESPONSES["get_weather"].get(location, {
            "status": "success",
            "current": {"temp": 30, "humidity": 70, "condition": "nắng"},
            "note": f"Thời tiết {location} hôm nay: nắng, nhiệt độ khoảng 30°C."
        })

    elif tool_name == "execute_robot_command":
        action = arguments.get("action")
        light_state = arguments.get("light_state")
        location_id = arguments.get("location_id")

        if action == "set_light":
            key = f"set_light_{light_state}"
            response = MOCK_RESPONSES["execute_robot_command"][key].copy()
            response["location_id"] = location_id
            return response

        return MOCK_RESPONSES["execute_robot_command"].get(action, {
            "status": "success",
            "message": f"Đã thực hiện {action} thành công."
        })

    elif tool_name == "search_web":
        return MOCK_RESPONSES["search_web"]["default"]

    elif tool_name == "query_knowledge_base":
        return MOCK_RESPONSES["query_knowledge_base"]["default"]

    return {"status": "success", "message": "Mock response"}


# Robot state for multi-turn testing
class RobotState:
    """Simulates robot state for multi-turn testing"""

    def __init__(self):
        self.lights = {
            "living_room": "off",
            "bedroom": "off",
            "kitchen": "off",
            "garden": "off"
        }
        self.robot_position = "living_room"

    def get_light_state(self, location_id: str) -> str:
        """Get current light state for a location"""
        return self.lights.get(location_id, "off")

    def set_light(self, location_id: str, state: str) -> dict:
        """Set light state and return response"""
        self.lights[location_id] = state
        return {
            "status": "success",
            "action": "set_light",
            "location_id": location_id,
            "light_state": state,
            "message": f"Đã {'bật' if state == 'on' else 'tắt'} đèn {location_id}."
        }

    def to_dict(self) -> dict:
        """Return current state as dict"""
        return {
            "lights": self.lights.copy(),
            "robot_position": self.robot_position
        }


# Global state instance for testing
_robot_state = RobotState()


def get_robot_state() -> RobotState:
    """Get the global robot state instance"""
    return _robot_state


def reset_robot_state():
    """Reset robot state to default"""
    global _robot_state
    _robot_state = RobotState()
