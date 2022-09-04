#!/bin/bash
echo "Let's set up a new user or change your old password"
echo -n "email address: "
read email
echo -n "password: "
read password
echo  $email $password

sqlite3 -line /home/beholder/Desktop/beholder-data/database.sql "DELETE FROM user WHERE email = '$email'" || echo "new user"
cd /home/beholder/beholder-device/beholder-recorder && .venv/bin/flask users create $email --password $password --active
