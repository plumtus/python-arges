#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Arges Server Client

Required packages:

pip install requests requests_oauthlib requests-toolbelt

pip install opencv-python
"""
import os
import sys
import re
import cv2
import numpy
import logging
import datetime
import json
import requests
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
from requests.auth import HTTPBasicAuth
from requests_toolbelt import MultipartEncoder

class ArgesError(Exception):
    def __init__(self, message, code = 0):
        self.message = message
        self.code = code
    def __str__(self):
        return repr(self.message)

class Gender:
    (Unknown, Male, Female) = range(0, 3)
    labels = ("Unknown", "Male", "Female")

class Emotion:
    (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) = range(0, 7)
    labels = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")

class Face:
    """
    Face object

        Attribtues:

            top, left, right, bottom (int): positions of face in original image
            pitch, roll, yaw (float): euler angles
            leftEye, rightEye, leftEyebrow, rightEyebrow, mouth, nose, jaw (list): landmarks in face
            gender (Gender): gender
            age (int): age
            emotion (Emotion): emotion
    """
    def __init__(self):
        self.id = ""
        self.region = None
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.leftEye = None
        self.rightEye = None
        self.mouth = None
        self.nose = None
        self.jaw = None
        self.leftEyebrow = None
        self.rightEyebrow = None
        self.gender = Gender.Unknown;
        self.age = (0, 0)
        self.emotion = Emotion.Neutral;
    def size(self):
        """ Return size of face """
        return (self.right - self.left, self.bottom - self.top)
    def position(self):
        """ Return the left-top coordinate of face """
        return (self.left, self.top)
    def __str__(self):
       info = "Face [%s]" % self.id
       if self.region != None:
           info += "\n  region (%d, %d, %d, %d)" % self.region
       info += "\n  angle (%0.2f, %0.2f, %0.2f)" % (self.pitch, self.yaw, self.roll)
       if self.leftEye != None:
           info += "\n  left-eye"
           for p in self.leftEye: info += (" (%d,%d)" % p)
       if self.rightEye != None:
           info += "\n  right-eye"
           for p in self.rightEye: info += (" (%d,%d)" % p)
       if self.leftEyebrow != None:
           info += "\n  left-eyebrow"
           for p in self.leftEyebrow: info += (" (%d,%d)" % p)
       if self.rightEyebrow != None:
           info += "\n  right-eyebrow"
           for p in self.rightEyebrow: info += (" (%d,%d)" % p)
       if self.nose != None:
           info += "\n  nose"
           for p in self.nose: info += (" (%d,%d)" % p)
       if self.mouth != None:
           info += "\n  mouth"
           for p in self.mouth: info += (" (%d,%d)" % p)
       if self.jaw != None:
           info += "\n  jaw"
           for p in self.jaw: info += (" (%d,%d)" % p)
       info += "\n  gender " + Gender.labels[self.gender]
       info += "\n  age %s" % repr(self.age)
       info += "\n  emotion " + Emotion.labels[self.gender]
       return info

class Group:
    def __init__(self, gid, name):
        self.id = gid
        self.name = name
        self.creation_time = None
        self.persons = []
    def __str__(self):
        info = "Group [%s] %s, creation time: %s" % \
               (self.id, self.name, self.creation_time.strftime("%Y/%m/%d %H:%M:%S"))
        for p in self.persons:
            info += "\n  [%s] %s" % p
        if len(self.persons) > 1:
            info += "Total %d persons" % len(self.persons)
        return info.encode("utf-8")

class Person:
    def __init__(self, pid, name):
        self.id = pid
        self.name = name
        self.code = ""
        self.gender = Gender.Unknown
        self.creation_time = None
        self.groups = []
        self.faces = []
        self.photo = None
    def __str__(self):
        info = "Person [%s] %s (%s) creation time: %s" % \
               (self.id, self.name, self.code, self.creation_time.strftime("%Y/%m/%d %H:%M:%S"))
        for f in self.faces:
            info += "\n  %s" % f
        if len(self.faces) > 1:
            info += "Total %d face images" % len(self.faces)
        for g in self.groups:
            info += "\n  group [%s] %s" % g
        if self.photo != None:
            info += "\n  " + self.photo
        return info.encode("utf-8")

class ArgesClient:
    def __init__(self, endpoint, app_id, app_key): 
        """
        Initialize ArgesClient using endpoint of server, app id and app key
        """
        if endpoint is None:
            self.endpoint = "http://127.0.0.1:8080/server";
        else:
            if endpoint[-1] == "/":
                self.endpoint = endpoint[:-1]
            else:
                self.endpoint = endpoint
        if bool(re.match("https://", self.endpoint, re.I)):
            pass
        elif bool(re.match("http://", self.endpoint, re.I)):
            os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
        else:
            raise ArgesError("Bad endpoint " + endpoint)
        if app_id is None or app_key is None:
            raise ArgesError("App ID and App key is required")
        self.app_id = app_id
        self.app_key = app_key
        self.access_token = None
        self.logger = logging.getLogger("arges")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s")
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.ERROR)
        self.ses = requests.Session()

    def setLoggerLevel(self, level):
        """ Set logging level """
        self.logger.setLevel(level)

    def detectFaces(self, image):
        """
        Detect faces from the input image

        Args:

            image (str or numpy.ndarray): image file name or image array (RGB)

        Returns: list of face
        """
        images = {"image-file": image}
        res = self._post("/face/detect", None, images)
        if res.has_key("faces") and len(res["faces"]) > 0:
            faces = []
            for idx,item in enumerate(res["faces"]):
                faces.append(self._assemble_face(item, idx));
            return faces
        else:
            return []

    def predictFeatures(self, image, emotion = False):
        """
        Detect and extract the faces from input image

        Args:

            image (str or numpy.ndarray): image file name or image array (RGB)
            features (bool): predict features from face

        Returns: list of face
        """
        images = {"image-file": image}
        params = {"feature": "true", "emotion" : "true" if emotion else "false" }
        res = self._post("/face/predict", params, images)
        if res.has_key("faces") and len(res["faces"]) > 0:
            faces = []
            for idx,item in enumerate(res["faces"]):
                faces.append(self._assemble_face(item, idx));
            return faces
        else:
            return []

    def compare(self, image1, image2):
        """
        Compare to face images and get the confidence of similarity.

        Args:

            image1 (str or numpy.ndarray): first image file name or image array (RGB)
            features (bool): predict features from face

        Returns: the confidence of similarity
        """
        images = {"image-file1": image1, "image-file2": image2}
        res = self._post("/compare", {}, images)
        return float(res["confidence"])

    def verify(self, pid, image, uniformed = False):
        """
        Verify the face in image is if the same person represented by the person ID.

        Args:

            pid (str): ID of person record
            image (str or numpy.ndarray): image file name or image array (RGB)
            uniformed (bool): If true indicate the image has the unimformed format (360x360, face centered)

        Returns: the confidence of similarity
        """
        params = {"person-id": pid, "uniformed": "true" if uniformed else "false"}
        images = {"image-file": image}
        res = self._post("/verify", params, images)
        return float(res["confidence"])

    def recognize(self, gid, image, uniformed = False):
        """
        Recognize the most similar person from persons in specified group.

        Args:

            gid (str): ID of group
            image (str or numpy.ndarray): image file name or image array (RGB)
            uniformed (bool): If true indicate the image has the unimformed format (360x360, face centered)

        Returns: Tuple (person, group, confidence), or None when no matched person was found.
        """
        params = {"group-id": gid, "uniformed": "true" if uniformed else "false"}
        images = {"image-file": image}
        try:
            res = self._post("/recognize", params, images)
            return (self._assemble_person(res["person"]), self._assemble_group(res["group"]), float(res["confidence"]))
        except ArgesError as e:
            if e.code == 1011:
                return None
            else:
                raise e

    def search(self, gid, image, tops = None, threshold = None, uniformed = False):
        """
        Search the similar persons from the specified group.

        Args:

            gid (str): ID of group
            image (str or numpy.ndarray): image file name or image array (RGB)
            tops (int): Maximum records of result seta (optional)
            threshold (float): Similarity filtering threshold
            uniformed (bool): If true indicate the image has the unimformed format (360x360, face centered)

        Returns: List of tuple(person, group, confidence)
        """
        params = {"group-id": gid, "uniformed": "true" if uniformed else "false"}
        if tops is not None and tops > 0:
            params["tops"] = tops
        if (threshold is not None) and (threshold > 0.0 and threshold <= 1.0):
            params["threshold"] = threshold
        images = {"image-file": image}
        res = self._post("/search", params, images)
        rs = []
        for item in res:
            rs.append((self._assemble_person(item["person"]), self._assemble_group(item["group"]), float(item["confidence"])))
        return rs

    def listGroups(self):
        """
        Retrieve the group list

        Reeturns: list of tuples
        """
        res = self._get("/group/list", None)
        if len(res) == 0:
            return []
        groups = []
        for item in res:
            groups.append((item["id"], item["name"])) 
        return groups

    def queryGroup(self, gid):
        """
        Query the detail of the specified group

        Args:

            gid (str): ID of group

        Returns: Group object
        """
        res = self._get("/group/query", {"group-id": gid})
        return _assemble_group(res)

    def createGroup(self, name):
        """
        Create a new group.

        Args:

            name (str): name of group

        Returns: ID of this new group
        """
        res = self._post("/group/create", {"group-name": name})
        if not res.has_key("group-id"):
            raise ArgesError("Invalid response")
        return res["group-id"]

    def deleteGroups(self, ids):
        """
        Delete a set of groups.

        Args:

            ids (list): List of group IDs

        Returns: number of group has been deleted
        """
        if ids is None or len(ids) == 0:
            return 0
        res = self._post("/group/delete", {"group-ids": ",".join(ids)})
        if not res.has_key("deleted"):
            raise ArgesError("Invalid response")
        return int(res["deleted"])

    def renameGroup(self, gid, name):
        """
        Change the name of the specified group.

        Args:

            gid (str): ID of group to rename
            name (str): New name of group
        """
        self._post("/group/edit", {"group-id" : gid, "group-name": name})

    def appendPersonsToGroup(self, groupID, personIDs):
        """
        Append a set of persons to the specified group.

        Args:

            groupID (str): ID of group to add to
            personIDs (list): List of person IDs to be added

        Returns: number of persons has been appended
        """
        if personIDs is None or len(personIDs) == 0:
            return 0
        res = self._post("/group/append-person", {"group-id" : groupID, "person-ids": ",".join(personIDs)})
        if not res.has_key("appended"):
            raise ArgesError("Invalid response")
        return int(res["appended"])

    def removePersonsFromGroup(self, groupID, personIDs):
        """
        Remove a set of persons from the specified group.

        Args:

            groupID (str): ID of group to remove from
            personIDs (list): List of person IDs to be removed

        Returns: number of persons has been removed
        """
        if personIDs is None or len(personIDs) == 0:
            return 0
        res = self._post("/group/remove-person", {"group-id" : groupID, "person-ids": ",".join(personIDs)})
        if not res.has_key("removed"):
            raise ArgesError("Invalid response")
        return int(res["removed"])

    def clearGroup(self, gid):
        """
        Remove all persons from the specified group.

        Args:

            gid (str): ID of group to clear

        Returns: number of persons has been removed
        """
        res = self._post("/group/clear", {"group-id" : gid})
        if not res.has_key("removed"):
            raise ArgesError("Invalid response")
        return int(res["removed"])

    def listPersons(self, gid):
        """
        Retrieve all persons in the specified group (or un-grouped persons).

        Args:

            gid (str): ID of group or None for un-grouped person

        Returns: List of person IDs
        """
        return self._get("/person/list", params = {"group-id" : gid} if gid != None else None)

    def queryPerson(self, pid = None, code = None):
        """
        Query the detail information of person according ID or code.

        Args:

            pid (str): ID of person
            code (str): Code of person

        Returns: Person object
        """
        if pid is not None:
            res = self._get("/person/query", {"person-id" : pid})
        elif code is not None:
            res = self._get("/person/query-code", {"person-code" : code})
        return self._assemble_person(res)

    def createPerson(self, name, code = None, imageIDs = [], portrait = None, groupIDs = []):
        """
        Create person record.

        Args:

            name (str): Name of person
            code (str): Unique code of person (optional, if None code will be auto generated)
            imageIDs (list): ID (result of detectFaces) list of face images (optional)
            portrait (str): One of imageIDs as portrait (optional)
            groupIDs (list): ID list of group to added to (optional)

        Returns: Person object
        """
        params = {"person-name" : name}
        if code != None and len(code) > 0:
            params["person-code"] = code
        if imageIDs != None and len(imageIDs) > 0:
            params["image-ids"] = ",".join(imageIDs)
        if portrait != None and len(portrait) > 0:
            params["portrait"] = portrait
        if groupIDs != None and len(groupIDs) > 0:
            params["group-ids"] = ",".join(groupIDs)
        res = self._post("/person/create", params)
        return self._assemble_person(res)

    def updatePerson(self, pid, name = None, code = None, imageIDs = [], portrait = None, override = True):
        """
        Create person record.

        Args:

            id (str): ID of person to be updated
            name (str): New name of person (optional)
            code (str): Unique code of person (optional)
            imageIDs (list): ID (result of detectFaces) list of face images (optional)
            portrait (str): One of imageIDs as portrait (optional)
            override (bool): Update mode: overwrite or append

        Returns: Person object
        """
        params = {"person-id" : pid}
        if name != None and len(name) > 0:
            params["person-name"] = name
        if code != None and len(code) > 0:
            params["person-code"] = code
        if imageIDs != None and len(imageIDs) > 0:
            params["image-ids"] = ",".join(imageIDs)
        if portrait != None and len(portrait) > 0:
            params["portrait"] = portrait
        params["override"] = "true" if override else "false"
        res = self._post("/person/update", params)
        return self._assemble_person(res)

    def deletePersons(self, ids):
        """
        Delete a set of persons.

        Args:

            ids (list): List of person IDs

        Returns: number of person has been deleted
        """
        if ids is None or len(ids) == 0:
            return 0
        res = self._post("/person/delete", {"person-ids": ",".join(ids)})
        if not res.has_key("deleted"):
            raise ArgesError("Invalid response")
        return int(res["deleted"])

    def appendFaceToPerson(self, pid, image, strict=True):
        """
        Append face image to person record

        Args:

            pid (str): ID of person to append to
            image (str or numpy.ndarray): file name or numpy image array (RGB)
            strict (bool): if true and multi-faces is found, exception will raised.
                           else using the first face image if multi-faces is found.

        Returns: Tuple of person ID and the ID of face image appended
        """
        params = {"person-id": pid, "strict": "true" if strict else "false"}
        images = {"image-file": image}
        res = self._post("/person/append-face", params, images=images)
        return (res["person-id"], res["faceId"])

    def removeFacesFromPerson(self, pid, faceIDs):
        """
        Remove face images from person record

        Args:

            pid (str): ID of person to append to
            faceIDs (list): List of face IDs to be removed

        Returns: Number of faces has been removed
        """
        if faceIDs is None or len(faceIDs) == 0:
            return 0
        res = self._post("/person/remove-face", {"person-id": pid, "face-ids": ",".join(faceIDs)})
        if not res.has_key("removed"):
            raise ArgesError("Invalid response")
        return int(res["removed"])

    def clearFacesFromPerson(self, pid):
        """
        Remove all face images from person record

        Args:

            pid (str): ID of person

        Returns: Number of face images has been removed
        """
        res = self._post("/person/clear", {"person-id": pid})
        if not res.has_key("removed"):
            raise ArgesError("Invalid response")
        return int(res["removed"])

    def refreshPerson(self, pid):
        """
        Refresh the person record to searchable and updated in searching engine
        when this person record is created or face images changed.

        Args:

            pid (str): ID of person to refresh
        """
        self._post("/person/refresh", {"person-id": pid})

    def _get(self, path, params):
        self._fetch_token()
        fields = {}
        if params != None and len(params) > 0:
            fields.update(params)
        h = {"Authorization" : "Bearer " + self.access_token}
        res = self.ses.get(self.endpoint + "/api" + path, params = fields, headers = h)
        if res.status_code == 200:
            if len(res.content) == 0:
                return None
            try:
                return res.json()
            except ValueError as err:
                raise ArgesError(str(err))
        elif res.status_code == 500:
            err = res.json()
            raise ArgesError(err["message"], code = err["code"])
        else:
            raise ArgesError("HTTP Error: %d" % res.status_code)

    def _post(self, path, params, images = None):
        self._fetch_token()
        fields = {}
        if params != None and len(params) > 0:
            fields.update(params)
        if images != None and len(images) > 0:
            for k, v in images.items():
                fields[k] = self._get_image(v)
        encoder = MultipartEncoder(fields = fields)
        h = {'Content-Type': encoder.content_type, "Authorization" : "Bearer " + self.access_token}
        res = self.ses.post(self.endpoint + "/api" + path, data = encoder, headers = h)
        if res.status_code == 200:
            if len(res.content) == 0:
                return None
            try:
                return res.json()
            except ValueError as err:
                raise ArgesError(str(err))
        elif res.status_code == 500:
            err = res.json()
            raise ArgesError(err["message"], code = err["code"])
        else:
            raise ArgesError("HTTP Error: %d" % res.status_code)

    def _get_image(self, image):
        if isinstance(image, str):
           if bool(re.match('.+\.jpg$', image, re.I)):
               with open(image, "rb") as fp:
                   return (os.path.basename(image), fp.read(), "image/jpeg")
           elif bool(re.match('.+\.png$', image, re.I)):
               with open(image, "rb") as fp:
                   return (os.path.basename(image), fp.read(), "image/png")
        elif isinstance(image, numpy.ndarray):
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _, img = cv2.imencode(".jpg", bgr)
            return ("image", img.tobytes(), "image/jpeg")

    def _fetch_token(self, force = False):
        if (not self.access_token is None) and (not force):
            return
        auth = HTTPBasicAuth(self.app_id, self.app_key)
        app_client = BackendApplicationClient(client_id = self.app_id)
        oauth = OAuth2Session(client = app_client)
        try:
            token = oauth.fetch_token(token_url = self.endpoint + "/oauth/token", auth = auth)
        except Exception as err:
            self.access_token = None
            self.logger.error("OAuth failed: " + str(err))
            raise ArgesError("OAuth failed: " + str(err))
        else:
            self.access_token = token["access_token"]
            if self.access_token is None:
                raise ArgesError("Unauthorized")
            self.ses.headers.update({"Authorization" : "Bearer " + self.access_token})
            self.expires_in = datetime.timedelta(seconds = token["expires_in"])
            self.expires_at = datetime.datetime.fromtimestamp(token["expires_at"])
            self.logger.debug("OAuth success: " + self.access_token)
            self.logger.debug("  expires in %d seconds" % self.expires_in.total_seconds())
            self.logger.debug("  expires at %s" % self.expires_at.strftime("%Y-%m-%d %H:%M:%S"))

    def _assemble_face(self, obj, idx):
        f = Face()
        if obj.has_key("image-id") and len(obj["image-id"]) > 0:
            f.id = obj["image-id"]
        else:
            f.id = str(idx)
        if obj.has_key("range"):
            r = obj["range"]
            f.region = (int(r["x"]), int(r["y"]), int(r["x"]) + int(r["width"]), int(r["y"]) + int(r["height"]))
        if obj.has_key("pitch"): f.pitch = float(obj["pitch"])
        if obj.has_key("roll"): f.roll = float(obj["roll"])
        if obj.has_key("ray"): f.ray = float(obj["ray"])
        if obj.has_key("left-eye") and len(obj["left-eye"]) > 0:
            f.leftEye = tuple((int(x["x"]), int(x["y"])) for x in obj["left-eye"])
        if obj.has_key("right-eye") and len(obj["right-eye"]) > 0:
            f.rightEye = tuple((int(x["x"]), int(x["y"])) for x in obj["right-eye"])
        if obj.has_key("left-eyebrow") and len(obj["left-eyebrow"]) > 0:
            f.leftEyebrow = tuple((int(x["x"]), int(x["y"])) for x in obj["left-eyebrow"])
        if obj.has_key("right-eyebrow") and len(obj["right-eyebrow"]) > 0:
            f.rightEyebrow = tuple((int(x["x"]), int(x["y"])) for x in obj["right-eyebrow"])
        if obj.has_key("nose") and len(obj["nose"]) > 0:
            f.nose = tuple((int(x["x"]), int(x["y"])) for x in obj["nose"])
        if obj.has_key("mouth") and len(obj["mouth"]) > 0:
            f.mouth = tuple((int(x["x"]), int(x["y"])) for x in obj["mouth"])
        if obj.has_key("jaw") and len(obj["jaw"]) > 0:
            f.jaw = tuple((int(x["x"]), int(x["y"])) for x in obj["jaw"])
        if obj.has_key("gender"): f.gender = Gender.labels.index(obj["gender"]) 
        if obj.has_key("age") and len(obj["age"]) > 0: f.age = tuple(int(x) for x in obj["age"])
        if obj.has_key("emotion"): f.emotion = Emotion.labels.index(obj["emotion"]) 
        return f

    def _assemble_group(self, obj):
        g = Group(obj["group"], obj["name"])
        if obj.has_key("creation"):
            g.creation_time = datetime.datetime.strptime(obj["creation"], "%Y-%m-%d %H:%M:%S")
        if obj.has_key("persons") and len(obj["persons"]) > 0:
            for item in obj["persons"]:
                g.persons.append((item["id"], item["name"]))
        return g

    def _assemble_person(self, obj):
        if (not obj.has_key("person")) or (not obj.has_key("name")):
            raise ArgesError("Invalid response")
        p = Person(obj["person"], obj["name"])
        if obj.has_key("code"):
            p.code = obj["code"]
        if obj.has_key("creation"):
            p.creation_time = datetime.datetime.strptime(obj["creation"], "%Y-%m-%d %H:%M:%S")
        if obj.has_key("gender"):
            p.gender = Gender.labels.index(obj["gender"])
        if obj.has_key("photo"):
            p.photo = obj["photo"]
        if obj.has_key("groups") and len(obj["groups"]) > 0:
            for item in obj["groups"]:
                p.groups.append((item["id"], item["name"]))
        if obj.has_key("faces") and len(obj["faces"]) > 0:
            p.faces = obj["faces"]
        return p

from skimage import io
if __name__ == "__main__":
    """ Example """
    ac = ArgesClient("http://localhost:8080/server", "arges", "341cf901261b0e1b")
    try:
        rs = ac.search("21ad5be0d2i1", sys.argv[1])
        for idx,r in enumerate(rs):
            print "[%d]" % idx, "-" * 20
            print "  " + str(r[0])
            print "  " + str(r[1])
            print "  " + str(r[2])
    except ArgesError as err:
        print err
