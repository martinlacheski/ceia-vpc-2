#!/bin/sh
# Genera la configuración S3 de SeaweedFS con las credenciales del entorno y arranca el servidor.
set -eu
sed -e "s|\${S3_ACCESS_KEY}|${S3_ACCESS_KEY}|" -e "s|\${S3_SECRET_KEY}|${S3_SECRET_KEY}|" \
  /etc/seaweedfs/s3.json.template > /tmp/s3.json
exec weed server -dir=/data -s3 -s3.port=8333 -s3.config=/tmp/s3.json
