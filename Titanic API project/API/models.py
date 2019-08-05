from ..create_db import db
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from marshmallow import fields, Schema


class PassengersModel(db.Model):
    __tablename__ = 'passengers'


