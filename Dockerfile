# vektor-server image: build the UI in a Node image and the server in the GCC
# image, then copy the binary and the UI's static files into a minimal
# "distroless" image (no shell or package manager, runs as non-root).
# Ollama is not included: it runs natively on the host.

FROM node:24.21.0-slim@sha256:0e0ff40c39bc087845bfb27465a0df4ea419520094bc35842ff83dd8cbe6f9b6 AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ui/.npmrc ./
# Exactly the locked versions; .npmrc stops packages from running install scripts.
RUN npm ci --no-fund
COPY ui ./
RUN npm run build

FROM gcc:14@sha256:9188ac751ca24431dc43dbd142a223c98ea74f01d2858e84d30ba342a0d67844 AS build
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY src ./src
# libstdc++ and libgcc are linked into the binary, so the run image only needs glibc.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DVEKTOR_TESTS=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    && cmake --build build --target vektor-server -j \
    && strip build/vektor-server \
    && mkdir /data

# glibc only: no libstdc++ (it is in the binary) and no OpenSSL (not used).
FROM gcr.io/distroless/base-nossl-debian13:nonroot@sha256:8c563c1fb5e120606f0d85733049775faed6192e2bd2223ef283a5393eec22b9
COPY --from=build /src/build/vektor-server /usr/local/bin/vektor-server
COPY --from=build --chown=65532:65532 /data /data
COPY --from=ui /ui/dist /ui
EXPOSE 8080
# Extra arguments after the image name are added to these, e.g. --llm-model qwen2.5:1.5b.
ENTRYPOINT ["/usr/local/bin/vektor-server", "--host", "0.0.0.0", "--port", "8080", \
            "--data-dir", "/data", "--ollama", "http://host.docker.internal:11434", "--ui", "/ui"]
