#!/bin/bash
set -x

# install ubuntu libraries
add-apt-repository ppa:gstreamer-developers/ppa
apt install libgstreamer* gstreamer* pulseaudio* \
            sqlite3 python3.9-dev python3.9-venv libpython3.9-dev virtualenv

# update sudoers so that the device can restart without root access
echo "allowing beholder to reboot device"
RESTART_LINES='%users  ALL=(ALL) NOPASSWD: /sbin/poweroff, /sbin/reboot'
grep -qxF "$RESTART_LINES" /etc/sudoers || echo "\n#auto-generated-by-beholder\n#Allow any user to reboot\n${RESTART_LINES}" >> /etc/sudoers

# create config directory
echo "creating config directory"
mkdir --mode=0755 -p /srv/beholder-config
sudo chown -R beholder:beholder /srv/beholder-config

# install services
echo "installing recording & webapp service"
cp ops/webapp.service /etc/systemd/system
systemctl enable webapp
systemctl start webapp

cp ops/recorder.service /etc/systemd/system
systemctl enable recorder
systemctl start recorder
