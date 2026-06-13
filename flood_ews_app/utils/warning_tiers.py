def get_tier(probability: float) -> dict:
    """Map a flood probability [0,1] to a PAGASA-style warning tier."""
    if probability < 0.25:
        return {
            "tier_name": "Green",
            "color": "#2ecc71",
            "text_color": "#ffffff",
            "icon": "🟢",
            "recommended_action": "No action needed. Monitor conditions normally.",
        }
    elif probability < 0.50:
        return {
            "tier_name": "Yellow",
            "color": "#f1c40f",
            "text_color": "#333333",
            "icon": "🟡",
            "recommended_action": "Stay alert. Monitor weather updates closely.",
        }
    elif probability < 0.75:
        return {
            "tier_name": "Orange",
            "color": "#e67e22",
            "text_color": "#ffffff",
            "icon": "🟠",
            "recommended_action": (
                "Prepare for possible evacuation. Secure belongings and "
                "move to higher ground if near waterways."
            ),
        }
    else:
        return {
            "tier_name": "Red",
            "color": "#e74c3c",
            "text_color": "#ffffff",
            "icon": "🔴",
            "recommended_action": (
                "Evacuate immediately if in a flood-prone area. "
                "Follow LGU instructions."
            ),
        }
