BUILD = build
MAIN  = main

.PHONY: all clean

all:
	mkdir -p $(BUILD)/sections
	xelatex -interaction=nonstopmode -output-directory=$(BUILD) $(MAIN).tex
	xelatex -interaction=nonstopmode -output-directory=$(BUILD) $(MAIN).tex

clean:
	rm -rf $(BUILD)
