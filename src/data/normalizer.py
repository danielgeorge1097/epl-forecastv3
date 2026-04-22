from __future__ import annotations

from typing import Dict, Optional
import pandas as pd


DEFAULT_TEAM_MAP: Dict[str, str] = {
    # Manchester clubs
    "Man United": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester United": "Manchester United",

    "Man City": "Manchester City",
    "Manchester City": "Manchester City",

    # London / common aliases
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",

    "QPR": "Queens Park Rangers",
    "Queens Park Rangers": "Queens Park Rangers",

    "West Ham": "West Ham United",
    "West Ham United": "West Ham United",

    # Midlands / north / common short names
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",

    "Leeds": "Leeds United",
    "Leeds United": "Leeds United",

    "Leicester": "Leicester City",
    "Leicester City": "Leicester City",

    "Birmingham": "Birmingham City",
    "Birmingham City": "Birmingham City",

    "Blackburn": "Blackburn Rovers",
    "Blackburn Rovers": "Blackburn Rovers",

    "Bolton": "Bolton Wanderers",
    "Bolton Wanderers": "Bolton Wanderers",

    "Derby": "Derby County",
    "Derby County": "Derby County",

    "Bradford": "Bradford City",
    "Bradford City": "Bradford City",

    "Coventry": "Coventry City",
    "Coventry City": "Coventry City",

    "Ipswich": "Ipswich Town",
    "Ipswich Town": "Ipswich Town",

    "Charlton": "Charlton Athletic",
    "Charlton Ath": "Charlton Athletic",
    "Charlton Athletic": "Charlton Athletic",

    "Wigan": "Wigan Athletic",
    "Wigan Athletic": "Wigan Athletic",

    "Luton": "Luton Town",
    "Luton Town": "Luton Town",

    "Norwich": "Norwich City",
    "Norwich City": "Norwich City",

    "Cardiff": "Cardiff City",
    "Cardiff City": "Cardiff City",

    "Swansea": "Swansea City",
    "Swansea City": "Swansea City",

    "Hull": "Hull City",
    "Hull City": "Hull City",

    "Stoke": "Stoke City",
    "Stoke City": "Stoke City",

    "Brighton": "Brighton and Hove Albion",
    "Brighton & Hove Albion": "Brighton and Hove Albion",
    "Brighton and Hove Albion": "Brighton and Hove Albion",

    # Sheffield clubs
    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",

    "Sheff Wed": "Sheffield Wednesday",
    "Sheffield Wednesday": "Sheffield Wednesday",

    # Other clubs
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Newcastle United": "Newcastle United",

    "West Brom": "West Bromwich Albion",
    "West Bromwich Albion": "West Bromwich Albion",

    "Middlesbrough": "Middlesbrough",
    "Sunderland": "Sunderland",
    "Southampton": "Southampton",
    "Everton": "Everton",
    "Liverpool": "Liverpool",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Aston Villa": "Aston Villa",
    "Crystal Palace": "Crystal Palace",
    "Oldham Athletic": "Oldham Athletic",
    "Swindon Town": "Swindon Town",
    "Barnsley": "Barnsley",
    "Wimbledon": "Wimbledon",
    "Nottingham Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Portsmouth": "Portsmouth",
    "Reading": "Reading",
    "Fulham": "Fulham",
    "Bournemouth": "Bournemouth",
    "Watford": "Watford",
    "Burnley": "Burnley",
    "Brentford": "Brentford",
}
    

def canonical_team_name(team: object, team_map: Optional[Dict[str, str]] = None) -> str:
    if pd.isna(team):
        return "UNKNOWN_TEAM"

    mapping = team_map or DEFAULT_TEAM_MAP
    raw = str(team).strip()
    return mapping.get(raw, raw)


def season_string_to_end_year(value: object):
    if pd.isna(value):
        return None

    if isinstance(value, int):
        return int(value)

    text = str(value).strip()

    if text.isdigit() and len(text) == 4:
        return int(text)

    for sep in ["/", "-"]:
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()

                if left.isdigit() and len(left) == 4:
                    start_year = int(left)

                    if right.isdigit() and len(right) == 2:
                        century = start_year // 100
                        return century * 100 + int(right)

                    if right.isdigit() and len(right) == 4:
                        return int(right)

    return None