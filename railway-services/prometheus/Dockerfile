FROM prom/prometheus:v2.51.0

# Run as root to avoid volume permission issues on Railway
USER root

# Copy custom config
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Ensure data directory is writable
RUN mkdir -p /prometheus && chown -R root:root /prometheus

EXPOSE 9090

ENTRYPOINT [ "/bin/prometheus" ]
CMD [ "--config.file=/etc/prometheus/prometheus.yml", \
      "--storage.tsdb.path=/prometheus", \
      "--storage.tsdb.retention.time=30d", \
      "--web.listen-address=0.0.0.0:9090", \
      "--web.enable-lifecycle" ]
