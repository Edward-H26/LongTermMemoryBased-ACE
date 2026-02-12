"""CL-bench v2 benchmark modules.

V2 replaces LangGraph with direct API calls and fixes the v1 message
preservation bug (assistant messages were incorrectly filtered out).

Modules
-------
infer_baseline  – GPT-5.1 baseline inference with multi-turn support.
infer_ace       – Direct-API ACE inference with Reflector + Curator.
"""
