FROM selenium/node-chrome:4.27.0

# Fly.io sets FLY_PRIVATE_IP to the machine's WireGuard mesh IPv6 address.
# Selenium nodes default to reporting their container-local 172.x.x.x IP,
# which is NOT routable from other Fly apps. This wrapper exports
# SE_NODE_HOST so the node registers with the hub using a reachable address.
COPY entrypoint.sh /opt/bin/fly-entrypoint.sh
USER root
RUN chmod +x /opt/bin/fly-entrypoint.sh
USER 1200
ENTRYPOINT ["/opt/bin/fly-entrypoint.sh"]
