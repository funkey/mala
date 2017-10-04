FROM funkey/gunpowder:v0.3-pre5
LABEL maintainer jfunke@iri.upc.edu

# install mala

# assumes that mala package directory is in build context (the complementary
# Makefile ensures that)
ADD mala /src/mala/mala
ENV PYTHONPATH /src/mala:$PYTHONPATH
