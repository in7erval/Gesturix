import datetime

import certifi
import pymongo

from config import logger

ca = certifi.where()

client = pymongo.MongoClient(
    "mongodb+srv://username:AIWBqQf8MzW0YCWd@cluster0.ac26z.mongodb.net/test?retryWrites=true&w=majority",
    ssl=True, tlsCAFile=ca)
db = client.diploma
keypoints_collection = db.keypoints

logger.debug(f"{client=}")


def landmarks_to_document(landmark):
    document = dict()
    for i, landmark in enumerate(landmark):
        document[f"key_{str(i)}"] = {"x": landmark.x, "y": landmark.y, "z": landmark.z}
    document["time"] = datetime.datetime.utcnow()
    return document


def save_landmarks_to_db(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if len(hand_landmarks.landmark) == 21:
                # for i, landmark in enumerate(hand_landmarks.landmark):
                #     xs[i].append(landmark.x)
                #     ys[i].append(landmark.y)
                #     zs[i].append(landmark.z)
                # if len(nums) == 0:
                #     nums.append(0)
                # else:
                #     nums.append(nums[-1] + 1)
                # logging.debug("SAVE")
                doc = landmarks_to_document(hand_landmarks.landmark)
                keypoints_collection.insert_one(doc)
