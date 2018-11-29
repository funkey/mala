FROM funkey/gunpowder:v0.3
LABEL maintainer jfunke@iri.upc.edu

# install dependencies

RUN apt-get update && apt-get install -y --no-install-recommends \
        libmlpack-dev && \
    rm -rf /var/lib/apt/lists/*

# install mala

# assumes that mala package directory is in build context (the complementary
# Makefile ensures that)
ADD mala /src/mala/mala
ADD setup.py /src/mala/
WORKDIR /src/mala
RUN python setup.py install

RUN pip install mahotas
