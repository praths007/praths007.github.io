## Table of contents

- [CI CD and continuous deployment terminology](#CI-CD-and-continuous-deployment-terminology)
- [Integrate jenkins server with github](#integrate-jenkins-server-with-github)
- [Creating pipelines](#creating-pipelines)

## Integrate jenkins server with github (simpler version without pipeline)
This [video](https://youtu.be/Z3S2gMBUkBo) explains things in detail. This will build the code after each commit.
The high level steps are as follows:
* Manage jenkins > manage plugins > search available github integration plugin (download all and install after restart).
* Create job on jenkins (freestyle project/pipeline)
* Setup project URL in jenkins
    * Branches can be used as well
* Build triggers with option Github hook trigger for GITscm polling (this will do stuff after changes in github are 
made).
* Go to github settings and configure webhooks. Add payload URL as jenkins installation URL/github-webhook.
    * Can configure secrets
    * Notify for just push event

## Creating pipelines
This [video](https://youtu.be/s73nhwYBtzE) explains creation of pipelines. High level steps are as follows:
* Select pipeline and configure Jenkinsfile either from pre built or from specify SCM file from your own repo.
* Different stages can be created if using with maven specify the version in jenkinsfile, then create integration tests in the stages within the jenkins file. The file is
a groovy script.

## CI CD and continuous deployment terminology
* Continuous integration is when feature branches are merged with the master branch creating build after each commit
and automatically running tests against the build.
* Continuous delivery is an extension of CI where releases are made quickly and in small batches.
* Continuous deployment is a further extension all changes are released to customer without human intervention. Only
a failed test will prevent the release. It is similar to continuous delivery except it happens automatically.

