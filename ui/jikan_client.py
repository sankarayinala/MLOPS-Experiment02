import requests

JIKAN_URL = "https://api.jikan.moe/v4/anime/"

def get_poster_url(mal_id):
    try:
        resp = requests.get(f"{JIKAN_URL}{mal_id}")
        if resp.status_code != 200:
            return None

        return resp.json()["data"]["images"]["jpg"]["large_image_url"]
    except:
        return None