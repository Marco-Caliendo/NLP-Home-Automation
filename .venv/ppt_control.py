#!/usr/bin/env python3

import win32com.client
import pythoncom


class PPT:
    def __init__(self):
        self.ppt_app = None
        self.presentation = None  # Store the presentation object


    def open_presentation(self, path):
        # Set object to power point application
        self.ppt_app = win32com.client.Dispatch("PowerPoint.Application")
        # Open the presentation from the path
        self.presentation = self.ppt_app.Presentations.Open(path)
        # Make PowerPoint visible
        self.ppt_app.Visible = True
        # Show the presentation in full-screen mode
        self.presentation.SlideShowSettings.Run()


    def next_slide(self):
        # Get the SlideShow window object
        slideshow = self.ppt_app.SlideShowWindows(1)
        # Move through the slides
        slideshow.View.Next()  # Go to the next slide


    def prev_slide(self):
        # Get the SlideShow window object
        slideshow = self.ppt_app.SlideShowWindows(1)
        # Go to the previous slide
        slideshow.View.Previous()


    def end_presentation(self):
        # Close the presentation after the slideshow
        if self.presentation:
            self.presentation.Close()
            self.presentation = None
        # Quit PowerPoint
        self.ppt_app.Quit()
