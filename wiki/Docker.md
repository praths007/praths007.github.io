## Table of contents

- [Advantages of docker](#advantages-of-docker)
- [Difference between virtual machine and container](#difference-between-virtual-machine-and-container)
- [How it works](#how-it-works)

## Advantages of docker
* Same environment everywhere
* Sandbox projects

## Difference between virtual machine and container
* VM needs a kernel and container doesnt need kernel. So VM is more resource heavy for a host machine.
* Uses less memory and resources.
* Stacking of images means that creating a process doesnt need to start from scratch. Eg. to run jupyter, a pre
installed anaconda image will be available on docker hub.

## How it works
* Uses container which is the running instance of an ima.ge which is snapshot of a system at any given time.
    * It has got the OS, application codes softwares all bundled in a file.
* Images defined using Dockerfile
    * Text file with steps to create image, like install softeare, move files etc.

## How to install
### Simpler application using docker run
This [video](https://youtu.be/YFl2mCHdv24) contains good steps for a hello world application.
Steps to follow are as follows:
* Install docker for Mac/windows or Linux
* Create application folder src (eg. a python print hello world program)
* Create Dockerfile outside application folder
    * configure docker file
        * get a ready made image from docker hub (eg. official python image)
        * add the following to docker file
        ```bash
        FROM <specify image name from docker hub>
        COPY src/ /<folder path in image specified in docker hub>
        EXPOSE <port number>
        ```
* Build docker
```bash
docker build -t hello-world . <. means same directory>
```
* Launch docker using run
```bash
docker run -p 80:80 <guest:host mapping> 
```
Doing this will reflect changes made on host to guest.
```bash
docker run -p 80:80 -v <host path>:<guest path>
```
Endeavour to have one process per container. As if python crashes it takes the container with it.

### Microservices using docker compose
This [video](https://youtu.be/Qw9zlE3t8Ko) contains good steps for a hello world application.

        
      


