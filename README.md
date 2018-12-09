# meltsub

Takes a raw and an hardsub video, outputs subtitles extracted from the hardsub video.

## Disclaimer

I take no legal responsibility for anything this code is used for. This is purely an educational proof of concept.

## Usage

Dependencies:
* opencv
* tesseract

```shell
python meltsub.py
```

## Settings

Change the variable *subtitles_lang* to set another language.

Use this command to check installed language:

```shell
tesseract --list-langs
```

## License

MIT
