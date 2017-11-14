#!/bin/bash
python dumpTxt.py --split=train --answers=all
python dumpTxt.py --split=train --answers=modal
python dumpTxt.py --split=val --answers=all
python dumpTxt.py --split=val --answers=modal
