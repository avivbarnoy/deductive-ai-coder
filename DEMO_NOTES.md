# Reviewer demo flow

The bundled `sample_project.json` is a synthetic demonstration project.

For that project only, Stage 2 automatically preloads synthetic calibration cases so a reviewer can run the human–AI comparison immediately. Real researcher projects still start with a blank calibration table.

Stage 3 selects likely qualitative-text columns using common names such as `response`, `text`, `answer`, and `comment`. The built-in demo dataset selects `response` explicitly. The interface also warns when an ID-like column is selected and displays the actual text column used for a completed run.
