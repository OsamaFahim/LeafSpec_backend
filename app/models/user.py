from app import mongo
import secrets
import string
from werkzeug.security import generate_password_hash, check_password_hash

class User:
    def __init__(self, name=None, email=None, password=None, auth_type='Email'):
        self.name = name
        self.email = email
        self.password = password
        self.favourites = []
        self.auth_type = auth_type
        self.db = mongo['LeafSpec']

    def toJson(self):
        return {
            "name": self.name,
            "email": self.email,
            "favourites": self.favourites
            }

    @staticmethod
    def exists(email):
        user = mongo['LeafSpec'].users.find_one({"email": email})
        return user is not None
    
    @staticmethod
    def find_by_email(email):
        return mongo['LeafSpec'].users.find_one({"email": email})
    
    @staticmethod
    def find_by_id(id):
        return mongo['LeafSpec'].users.find_one({"_id": id})

    def save(self):
        if not self.name or not self.email or not self.password:
            raise ValueError("Name, email and password are required")
        
        hashed_password = generate_password_hash(self.password)
        user_id = self.db.users.insert_one({
            "name": self.name,
            "email": self.email,
            "password": hashed_password
        }).inserted_id
        return str(user_id)
    
    @staticmethod
    def update_favourites(email, favourites):
        mongo['LeafSpec'].users.update_one({"email": email}, {"$set": {"favourites": favourites}})

    def getUserFavourites(self):
        user = self.db.users.find_one({"email": self.email})
        return user['favourites']

    @staticmethod
    def authenticate(email, password):
        user = mongo['LeafSpec'].users.find_one({"email": email})
        if user and check_password_hash(user['password'], password):
            return user
        return None
    

    def generate_random_password(length=16):
        """
        Generate a cryptographically secure random password.
        
        Args:
            length (int): Length of the password. Defaults to 16.
        
        Returns:
            str: Randomly generated password
        """
        # Characters to choose from
        alphabet = string.ascii_letters + string.digits + string.punctuation
        
        # Generate password using cryptographically secure method
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        return password