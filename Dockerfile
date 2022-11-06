FROM rhub/r-minimal
RUN apk update
RUN installr -d -t gfortran Rcpp RcppEigen
RUN installr -d -t gsl-dev RcppGSL
RUN installr -d digest
RUN apk add --no-cache valgrind
RUN apk add --no-cache gsl-dev gcc musl-dev g++
CMD [ "sh" ]