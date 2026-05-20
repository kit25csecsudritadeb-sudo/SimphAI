# SimphAI Makefile

# Run the server
run:
	python3 -m http.server 8000

# Create zip for submission
zip:
	rm -f SimphAI.zip
	zip -r SimphAI.zip index.html main.js styles.css SimphAI_Presentation.md README.md Makefile

# Clean up
clean:
	rm -f SimphAI.zip session_log.json
