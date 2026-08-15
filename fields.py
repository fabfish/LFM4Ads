"""Feature field definitions.

``user`` / ``video`` map each field name to its embedding vocabulary size.
The hard-coded values below are the **KuaiRand-1K** vocabularies (the default).

For other datasets (e.g. KuaiRand-27K), set the ``LFM_VOCAB_JSON`` environment
variable to a JSON file of the form::

    {"user": {"user_id": 27000, ...}, "video": {"video_id": 4000000, ...}}

Only the *vocabulary sizes* are overridden — the field names (keys) are fixed
and shared by every dataset, so ``fields.all`` (the column order) never changes.
The override is applied before ``all`` is computed.
"""

import json
import os

user = {
    "user_id": 1000,
    "user_active_degree": 7,
    "is_lowactive_period": 1,
    "is_live_streamer": 2,
    "is_video_author": 2,
    "follow_user_num_range": 8,
    "fans_user_num_range": 8,
    "friend_user_num_range": 7,
    "register_days_range": 7,
    "onehot_feat0": 2,
    "onehot_feat1": 7,
    "onehot_feat2": 23,
    "onehot_feat3": 394,
    "onehot_feat4": 14,
    "onehot_feat5": 5,
    "onehot_feat6": 3,
    "onehot_feat7": 37,
    "onehot_feat8": 283,
    "onehot_feat9": 7,
    "onehot_feat10": 5,
    "onehot_feat11": 3,
    "onehot_feat12": 3,
    "onehot_feat13": 3,
    "onehot_feat14": 3,
    "onehot_feat15": 3,
    "onehot_feat16": 3,
    "onehot_feat17": 3,
}
video = {
    "video_id": 4369953,
    "author_id": 1407453,
    "video_type": 3,
    "upload_dt": 236,
    "upload_type": 32,
    "visible_status": 4,
    "music_id": 2621019,
    "music_type": 7,
    "tag": 1189,
}

# ---- optional dataset override (e.g. KuaiRand-27K) ---------------------
_override = os.environ.get("LFM_VOCAB_JSON")
if _override and os.path.exists(_override):
    with open(_override, "r", encoding="utf-8") as _f:
        _v = json.load(_f)
    user.update(_v.get("user", {}))
    video.update(_v.get("video", {}))

all = list(user | video) + ["date", "is_click", "tab"]
