DAY ?=1
# folder formatted as day002 
DAY_NUM := $(shell printf "%03d" $(DAY))

# Define the build command
.PHONY: build
build:
	$(MAKE) -C day$(DAY_NUM)

clean:
	$(MAKE) -C day$(DAY_NUM) clean

# A pattern rule to catch '1', '2', etc. from the command line
%:
	$(MAKE) build DAY=$@

