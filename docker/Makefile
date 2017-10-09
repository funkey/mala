TAG="funkey/mala:v0.1-pre1"

.PHONY: default push
default:
	-docker pull funkey/gunpowder:v0.3-pre5
	-docker rmi -f $(TAG)
	-rm -rf mala
	cp -r ../mala .
	cp ../setup.py .
	docker build -t $(TAG) .

push: default
	docker push $(TAG)
