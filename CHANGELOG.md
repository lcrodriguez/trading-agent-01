# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

- feat(ha_sma): replace SMA9/SMA50 golden-cross re-entry with price-recovery entry — re-enters when price closes >2% above SMA21 and above SMA50, fixing 100-200+ day out-of-market gaps during bull runs
- feat(ha_sma): add SMA9 cross-down exit guarded by SMA200 — exits when SMA9 crosses below SMA21 or SMA200, only when SMA21 is within 2% of SMA200, preventing premature sells in strong uptrends (replaces SMA130 breakdown signal)
