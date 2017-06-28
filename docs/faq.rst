Frequently Asked Questions
==========================

Below are answers to some common questions asked about ELFI.

*Q: My uniform prior* ``elfi.Prior('uniform', 1, 2)`` *does not seem to be right as it
produces outputs from the interval (1, 3).*

**A**: The distributions defined by strings are those from ``scipy.stats`` and follow
their definitions. There the uniform distribution uses the location/scale definition, so
the first argument defines the starting point of the interval and the second its length.

