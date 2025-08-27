DAY ?= 1
# folder formatted as day002 
DAY_NUM := $(shell printf "%03d" $(DAY))

# Build the specified day, or the default
.PHONY: build
build:
	$(MAKE) -C day$(DAY_NUM)

.PHONY: all
all:
	$(MAKE) -C day$(DAY_NUM) all

# Clean the specified day, or the default
.PHONY: clean
clean:
	$(MAKE) -C day$(DAY_NUM) clean
