INTERNAL_USER_ID_FIELD = "internal_user_id"
INTERNAL_ITEM_ID_FIELD = "internal_item_id"

map_user_item_text = {
    "item.title": "t",
    "item.category": "c",
    "item.genres": "g",
    "item.description": "d",
    "interaction.summary": "s",
    "interaction.reviewText": "r",
    "interaction.review": "r",
    "interaction.review_text": "r",
}

reverse_map_user_item_text = {
    "Amazon": {
        "t": "item.title",
        "c": "item.category",
        "d": "item.description",
        "s": "interaction.summary",
        "r": "interaction.reviewText",
    },
    "GR_UCSD": {
        "t": "item.title",
        "g": "item.genres",
        "d": "item.description",
        "r": "interaction.review_text"
    },
    "CGR": {
        "t": "item.title",
        "g": "item.genres",
        "d": "item.description",
        "r": "interaction.review"
    },
}


def get_profile(dataset, shortened):
    global reverse_map_user_item_text
    return [reverse_map_user_item_text[dataset][i] for i in list(shortened)]
