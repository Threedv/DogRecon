{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "canine_start = 10\n",
    "for i in range(144):\n",
    "    if i <=canine_start:\n",
    "        s = 2 - i/canine_start\n",
    "    elif canine_start < i <= canine_start+36:\n",
    "        s = 1\n",
    "    elif canine_start+36 < i <= canine_start*2 +36:\n",
    "        s = 1+ (i-canine_start)/canine_start\n",
    "    elif canine_start*2 +36 < i <= 108-2*canine_start:\n",
    "        s = 2\n",
    "    elif 108-2*canine_start < i <= 108-canine_start:\n",
    "        s = 2 -(i-(108-2*canine_start))/canine_start\n",
    "\n",
    "    elif 108-canine_start < i <= 144-canine_start:\n",
    "        s = 1\n",
    "    elif 144-canine_start < i <= 144:\n",
    "        s = 1+ (i-144+canine_start)/canine_start\n",
    "\n",
    "\n",
    "    print(s)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
