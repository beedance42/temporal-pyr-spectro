### Preprocessing Tips for Custom Datasets

If you wish to build a pyramid from your own dataset, here are some important dataset preprocessing tips:

* Videos need to be in .mp4 format. Images can be in any standard image format which can be read by imageio, such as .jpg or .png.

* Insert an empty text file with a filename that ends in 'MISSING.txt' for each missing image or video. Use a consistent naming/numbering convention so that these placeholder files for missing data sort into the right order alongside the real data files.

* The code has been optimized for 24-hours-per-day or 16-hours-per-day of data coverage. It should be able to handle 8-hours-per-day and 12-hours-per-day as well, but those options have not been thoroughly tested.

* If the base temporal resolution for your dataset does not align neatly with our timescales, one option is to start at a lower timescale and add in 'MISSING.txt' files to fill in the 'missing' data points. For instance, if your base timescale is 2.5 hours (meaning, you have images that were taken 2.5 hours apart), you can start your pyramid using the 30-minute timescale as base. You would add four 'MISSING.txt' empty placeholder files between every actual image so that the images occur at regular 2.5 hour intervals. The effect of those 'extra' blank files will disappear in the upper levels of the pyramid.

* The `datasets.json` file will need to be updated with metadata specific to your custom dataset:
  - **Dataset**: (string) The name you will use to refer to your dataset when running a program, so that it finds the correct data in this json file.
  - **filepath_name**: (string) How your dataset is referred to in the image or video filenames (which might be different than the dataset name).
  - **dirpath**: (string) The absolute path to the directory where your source image or video data lives (for input into the pyramid program).
  - **form**: (string) "multi-video" or "images" depending on the format of your source data.
  - **baselvl**: (int) Pyramid starting level.
      - Level 0: 1 frame per 0.0333 secs (30 fps)
      - Level 1: 1 frame per 0.1667 secs (6 fps)
      - Level 2: 1 frame per 0.5 secs (2 fps)
      - Level 3: 1 frame per 1 sec (1 fps)
      - Level 4: 1 frame per 5 secs (0.2 fps)
      - Level 5: 1 frame per 15 secs
      - Level 6: 1 frame per 30 secs
      - Level 7: 1 frame per 60 secs (1 min)
      - Level 8: 1 frame per 5 mins
      - Level 9: 1 frame per 15 mins
      - Level 10: 1 frame per 30 mins
      - Level 11: 1 frame per 60 mins (1 hr)
      - Level 12: 1 frame per 2 hrs
      - Level 13: 1 frame per 4 hrs
      - Level 14: 1 frame per 12 hrs (varies, if vids span < 24 hrs)
      - Level 15: 1 frame per 24 hrs (1 day)
      - Level 16: 1 frame per 3 days
      - Level 17: 1 frame per 6 days
      - Level 18: 1 frame per 30 days
      - Level 19: 1 frame per 90 days
      - Level 20: 1 frame per 180 days
      - Level 21: 1 frame per 360 days (approx 1 frame per year)
      - Levels 22+ will go up by powers of 2 (1 frame per 2 'years', per 4 'years', etc)
  - **fps**: (int or float) Frames-per-second for your base level. This would be 30 for level 0, but for higher levels this number is not necessarily an integer so a float number can be entered that is a close approximation of the appropriate fraction. For instance, if the base level is Level 11 (1-hour timescale) the fps would be 1/3600 and you would approximate that by entering the float number 0.00027778.
  - **vid_hrs_1day**: (int) Choose 8, 12, 16, or 24, based on how many hours per day your dataset covers. For instance, if your data comes from a webcam that is only operational for 16 hours per day (such as 5am to 9pm), then you would enter 16. 
  - **frames_per_vid**: (int) For "images" form, choose 1; however, for "multi-video" form choose the number of frames in a single 'ideal' video. For instance, if you have 24 hours of data split into six 4-hour videos and those videos are all 30fps, then each video will have 30fps x 4hrs x 60mins x 60secs = 432,000 frames per video. However, if your data is split into 10-minute videos, they will each have 30fps x 10mins x 60secs = 18,000 frames per video.
  - **pyr_result_path**: (string) The absolute path for the directory where completed pyramid files will be saved.

* The InputDataOneDay class in `inputdata.py` has a `self.globstring` attribute. This gets used with the glob method to pull up a list of all input files, so the globstring needs to match the filename format of your files with wildcards used as needed. See https://python.readthedocs.io/en/latest/library/glob.html. When the `self.globstring` string is used with glob filename expansion, the resulting list should include the filenames of all of the files in your **dirpath**. There are some if-elif-else conditional statements already written to account for the variety of filenames and directory structures in our datasets. If your filenames don't fit into the existing logic, you will need to update the existing statements or insert a new elif clause for your custom dataset case. Note that the code currently assumes that year, month, and day information is included in filenames; however, this may not be strictly necessary as long as the filenames have some sort of numbering convention that allows them to sort in the correct temporal order. 

#### Share your results with us!
If you create a temporal pyramid from a custom dataset and would like to share your results with us, we would love to see them and link back out to your research or site if that makes sense. It would be great to eventually compile a gallery of pyramids.   
