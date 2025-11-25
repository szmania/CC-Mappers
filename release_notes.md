# Release Notes

## Features
- Added custom mappers and XML schemas for all mapper files.
- Added `num_guns` attribute to faction units.
- Added `submod_tag` to factions.
- Added `submod_addon_tag` to factions and cultures schemas.
- Allowed multiple `General` and `Knights` elements in faction definitions.
- Renamed `Time Period.xml` to `TimePeriod.xml`.
- Renamed `FireForgedEmpire` directory.
- Capitalized the `Terrains` folder.

## Bug Fixes
- Removed buggy units from Chinese and medieval playthroughs.
- Allowed elements in `Faction` to appear in any order in `titles.xsd`.
- Made `subculture` and `key` attributes optional in `factions.xsd`.
- Allowed `Heritage` tag to exist without a `Culture` tag in `cultures.xsd`.
- Corrected faction unit assignment description in `README.md`.

## Documentation
- Added comprehensive documentation for custom mappers.
- Updated `README.md` with details on mapper loading order.
- Updated `README.md` with information about the `tag.txt` file.
- Updated `README.md` with detailed descriptions for Cultures and Terrains.
