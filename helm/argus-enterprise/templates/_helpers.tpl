{{/*
Expand the name of the chart.
*/}}
{{- define "argus.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "argus.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "argus.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "argus.labels" -}}
helm.sh/chart: {{ include "argus.chart" . }}
{{ include "argus.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "argus.selectorLabels" -}}
app.kubernetes.io/name: {{ include "argus.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Brain labels
*/}}
{{- define "argus.brain.labels" -}}
{{ include "argus.labels" . }}
app.kubernetes.io/component: brain
{{- end }}

{{- define "argus.brain.selectorLabels" -}}
{{ include "argus.selectorLabels" . }}
app.kubernetes.io/component: brain
{{- end }}

{{/*
MCP labels
*/}}
{{- define "argus.mcp.labels" -}}
{{ include "argus.labels" . }}
app.kubernetes.io/component: mcp
{{- end }}

{{- define "argus.mcp.selectorLabels" -}}
{{ include "argus.selectorLabels" . }}
app.kubernetes.io/component: mcp
{{- end }}

{{/*
Selenium Grid labels
*/}}
{{- define "argus.selenium.labels" -}}
{{ include "argus.labels" . }}
app.kubernetes.io/component: selenium
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "argus.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "argus.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
PostgreSQL connection string
*/}}
{{- define "argus.postgresql.url" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:$(POSTGRES_PASSWORD)@{{ include "argus.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else }}
postgresql://{{ .Values.postgresql.external.username }}:$(POSTGRES_PASSWORD)@{{ .Values.postgresql.external.host }}:{{ .Values.postgresql.external.port }}/{{ .Values.postgresql.external.database }}
{{- end }}
{{- end }}

{{/*
Redis connection URL
*/}}
{{- define "argus.redis.url" -}}
{{- if .Values.redis.enabled }}
redis://:$(REDIS_PASSWORD)@{{ include "argus.fullname" . }}-redis-master:6379
{{- else }}
redis://:$(REDIS_PASSWORD)@{{ .Values.redis.external.host }}:{{ .Values.redis.external.port }}
{{- end }}
{{- end }}

{{/*
MinIO endpoint
*/}}
{{- define "argus.minio.endpoint" -}}
{{- if .Values.minio.enabled }}
{{ include "argus.fullname" . }}-minio:9000
{{- else }}
{{ .Values.minio.external.endpoint }}
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "argus.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ .name }}
{{- end }}
{{- end }}
{{- end }}
