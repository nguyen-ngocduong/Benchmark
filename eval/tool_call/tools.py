"""
Tool definitions for evaluation
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_gold_price",
            "description": "Retrieve current market gold prices or historical gold price trends in Vietnam. Supports querying by specific date range and provider/brand (e.g. SJC, PNJ, DOJI, Mi Hong). Use action='current' for real-time rates, or action='history' with from_date and to_date for historical records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["current", "history"],
                        "description": "The query mode: 'current' retrieves the latest real-time gold prices; 'history' retrieves historical price data over a date range."
                    },
                    "from_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Start date in 'YYYY-MM-DD' format (e.g. '2026-08-15'). Required when action is 'history'."
                    },
                    "to_date": {
                        "type": "string",
                        "format": "date",
                        "description": "End date in 'YYYY-MM-DD' format (e.g. '2026-08-20'). Required when action is 'history'."
                    },
                    "source": {
                        "type": "string",
                        "enum": ["SJC", "BTMC", "DOJI", "BTMH", "PNJ"],
                        "default": "SJC",
                        "description": "Filter by source: SJC, BTMC, DOJI, BTMH, PNJ."
                    }
                },
                "required": ["action"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up weather forecasts, current conditions, temperature ranges, humidity, and precipitation for locations in Vietnam. Only supports locations in Vietnam.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Location or city name in Vietnam to search weather for (e.g. 'Hà Nội', 'TP Hồ Chí Minh', 'Đà Nẵng', 'Hải Phòng')."
                    },
                    "date": {
                        "type": "string",
                        "description": "Omit or leave empty for current weather. For forecasts, provide the target/start date in YYYY-MM-DD format."
                    },
                    "date_end": {
                        "type": "string",
                        "description": "Optional forecast end date in YYYY-MM-DD format."
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Perform live web searches for current events, news, sports, general facts, and real-time data. Use category='news' for news and current affairs, category='sports' for sports information, and category='encyclopedia' for other factual topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query. For Vietnamese topics, preserve or write natural, fully-accented Vietnamese.",
                        "maxLength": 4000,
                        "minLength": 1
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["normal", "deep_research"],
                        "default": "normal"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["news", "sports", "encyclopedia"]
                    }
                },
                "required": ["query", "category"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "Search and retrieve authoritative documents from the internal knowledge base regarding VinDynamics, Vingroup ecosystem, butler robot Dyno specifications, features, hardware capabilities, smart home integration, and organizational information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Question, keywords, or search phrase to query against internal knowledge chunks."
                    },
                    "unknown_answer": {
                        "type": "string",
                        "default": "",
                        "description": "Optional fallback text to return if no relevant knowledge base document matches."
                    },
                    "limit_chunk": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                        "description": "Maximum number of top-matching document chunks to return."
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": []
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_robot_command",
            "description": "Execute a mock household robot command for movement, lights, area inspection, or calling an allowed family contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["move", "set_light", "inspect", "call"],
                        "description": "The household robot operation to perform."
                    },
                    "robot_id": {
                        "type": "string",
                        "default": "robot-demo-01"
                    },
                    "location_id": {
                        "type": "string",
                        "enum": ["floor_1", "floor_2", "floor_3", "living_room", "bedroom", "kitchen", "garden"]
                    },
                    "light_state": {
                        "type": "string",
                        "enum": ["on", "off"]
                    },
                    "contact_id": {
                        "type": "string",
                        "enum": ["family_mom", "family_dad", "family_older_brother", "family_older_sister", "family_younger_brother", "family_younger_sister"]
                    }
                },
                "required": ["action"],
                "additionalProperties": False
            }
        }
    }
]
