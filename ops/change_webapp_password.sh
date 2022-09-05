#!/bin/bash
echo "Changing your old password"
echo -n "email address: "
read email
echo -n "password: "
read password

database=$(awk -F "=" '/database/ {print $2}' /home/beholder/Desktop/beholder.ini)
repo=$(awk -F "=" '/repo/ {print $2}' /home/beholder/Desktop/beholder.ini)

sqlite3 -line $database "DELETE FROM user WHERE email = '$email'" && echo "resetting password"
cd $repo && cd beholder-recorder && .venv/bin/flask users create $email --password $password --active
