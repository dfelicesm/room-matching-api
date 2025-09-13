import pandas as pd

from src.matching import match_supplier_room


def test_no_matches_if_hotel_not_in_catalog():
    internal_df = pd.DataFrame(
        [
            {
                "lp_id": "lpAAAA",
                "room_name": "Twin Room with Balcony",
                "normalized": "twin room with balcony",
                "room_type": "room",
                "bed_type": None,
                "num_beds": None,
                "num_bedrooms": None,
            }
        ]
    )

    result = match_supplier_room(internal_df, "lpBBBB", "Room, 1 King Bed")
    assert result.empty


def test_matches_returned_for_known_hotel():
    """
    Ensure that when the supplier's room description is textually close
    enough (so that it passes the first filter) to an internal room for
    the same hotel, at least one match is returned with a similarity score
    between 0 and 100.
    """
    internal_df = pd.DataFrame(
        [
            {
                "lp_id": "lp42",
                "room_name": "Twin Room with Balcony",
                "normalized": "twin room with balcony",
                "room_type": "room",
                "bed_type": "twin",
                "num_beds": 2,
                "num_bedrooms": None,
            }
        ]
    )

    result = match_supplier_room(internal_df, "lp42", "Twin Room Balcony")

    assert not result.empty

    similarity_score = result.iloc[0]["similarity"]

    assert 0 <= similarity_score <= 100
