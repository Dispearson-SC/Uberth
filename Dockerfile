# The replay dashboard, served as what it is: a static page over static JSON.
#
# There is no application here. A shift is simulated offline, every tick is
# recorded to a file, and the browser plays it back with zero computation on
# the server -- which is the whole reason the demo cannot stall on stage. So
# this image is nginx and the repository, and nothing else.
FROM nginx:1.27-alpine

# Only what the page actually reads. The fixtures, the virtualenv and the
# Python that produced these recordings are not needed to show them, and
# leaving them out keeps the image at a few megabytes.
COPY replays/ /usr/share/nginx/html/replays/
COPY src/viz/ /usr/share/nginx/html/src/viz/

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
