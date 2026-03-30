FROM jenkins/jenkins:lts

USER root

# Install Docker CLI
RUN apt-get update -y && \
    apt-get install -y ca-certificates curl && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
      https://download.docker.com/linux/debian \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update -y && \
    apt-get install -y docker-ce-cli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Fix docker group GID to match host + add jenkins user
RUN DOCKER_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo 999) && \
    groupadd -g $DOCKER_GID -o docker || true && \
    usermod -aG docker jenkins

USER jenkins