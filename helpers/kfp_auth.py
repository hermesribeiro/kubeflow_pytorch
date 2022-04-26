import os
import requests

import kfp

HOST = os.environ['KFP_HOST']
USERNAME = os.environ['KFP_USERNAME']
PASSWORD = os.environ['KFP_PASSWORD']
NAMESPACE = os.environ['KFP_NAMESPACE']

class KfpAuth:
    """Authenticates to kfp server using login/password"""


    def __init__(self, host=None, username=None, password=None, namespace=None):
        self.host = (host or HOST)
        self.username = (username or USERNAME)
        self.password = (password or PASSWORD)
        self.namespace = (namespace or NAMESPACE)
        self.session_cookie = self._get_session_cookie()

    def _get_session_cookie(self):
        session = requests.Session()
        response = session.get(self.host)
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"login": self.username, "password": self.password}
        session.post(response.url, headers=headers, data=data)
        session_cookie = session.cookies.get_dict()["authservice_session"]
        return session_cookie

    def client(self):
        
        client = kfp.Client(
            host=f"{HOST}/pipeline",
            cookies=f"authservice_session={self.session_cookie}",
            namespace=NAMESPACE,
        )

        connected = bool(client.list_pipelines())
        print(f"Connection {'estabilished' if connected else 'failed'}")

        return client
