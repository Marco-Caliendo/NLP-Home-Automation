import win32com.client


# Start PowerPoint application
ppt_app = win32com.client.Dispatch("PowerPoint.Application")


class PPT:
    def __init__(self):
        self.presentation = None  # Store the presentation object


    # Open a PowerPoint presentation
    def open_presentation(self, path):
        self.presentation = ppt_app.Presentations.Open(path)
        # Make PowerPoint visible
        ppt_app.Visible = True
        # Show the presentation in full-screen mode
        self.presentation.SlideShowSettings.Run()


    def next_slide(self):
        # Get the SlideShow window object
        slideshow = ppt_app.SlideShowWindows(1)
        # Move through the slides
        slideshow.View.Next()  # Go to the next slide


    def prev_slide(self):
        # Get the SlideShow window object
        slideshow = ppt_app.SlideShowWindows(1)
        # Go to the previous slide
        slideshow.View.Previous()


    def end_presentation(self):
        # Close the presentation after the slideshow
        if self.presentation:
            self.presentation.Close()
        # Quit PowerPoint
        ppt_app.Quit()
