from flask_marshmallow import Schema
from marshmallow.fields import Str


class HomeSchema(Schema):
    class Meta:
        # Fields to expose
        fields = ["message"]
    message = Str()

#
