# Air-gapped GDK application sandbox
# NO NETWORK ACCESS - all outbound traffic is blocked
# NO HOST FILESYSTEM ACCESS beyond mounted game data (read-only)
# Base image pinned by digest; refresh deliberately (see sandbox/README.md).
FROM ubuntu:24.04@sha256:9c7f200c07e3d4a03e7249d2c16bf66a51120fb876aa2242cf31451412543289

# Install Wine, Vulkan, and vkd3d-proton (build-time only network)
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        wine64 wine32 \
        mesa-vulkan-drivers vulkan-tools \
        libvulkan1 libvulkan1:i386 \
        ca-certificates curl xvfb \
    && rm -rf /var/lib/apt/lists/*

# Non-root user whose uid/gid match the host user, so the bind-mounted prefix
# and logs directories are writable (ubuntu:24.04 ships "ubuntu" at 1000).
ARG UID=1000
ARG GID=1000
RUN (userdel -r ubuntu 2>/dev/null || true) && \
    groupadd -g ${GID} runner && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash runner
USER runner
WORKDIR /home/runner

# wine64/wine32 install their loaders in /usr/lib/wine; the /usr/bin/wine
# wrapper comes from the "wine" metapackage, which is not installed.
ENV PATH=/usr/lib/wine:$PATH
# Wine prefix will be initialized at runtime
ENV WINEPREFIX=/home/runner/.wine
ENV WINEDEBUG=-all
ENV DISPLAY=:99
