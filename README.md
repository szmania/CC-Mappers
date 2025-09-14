# CC-Mappers
Mappers for [Crusader Conflicts](https://github.com/szmania/Crusader-Wars/releases/latest), Community Successor for Crusader Wars mod.

## Mapper Loading Order

The Crusader Conflicts mod loads culture and faction mapping XML files in a specific order to allow for overrides. The loading process is as follows:

1.  **Base Mappers**: The official playthrough base mappers are loaded first. These files are identified by the naming convention `OfficialCC_<mapper_tag>_*.xml`.
2.  **Submod Mappers**: After the base mappers, all other XML files in the cultures and factions directories are loaded in alphabetical order. These are typically submod mappers.

This loading order means that any mapping defined in a submod mapper will overwrite a mapping for the same culture or faction from a base mapper. If multiple submod mappers define the same mapping, the one loaded last (alphabetically) will take precedence.

## License
The original source code and all further edits of this repository fall under the GNU GENERAL PUBLIC LICENSE Version 3 license of distribution, and it was copied from the v1.2.2 version of the CW-Mappers distributed here inside the Crusader-Wars release (which is GPL3 licensed) https://github.com/farayC/Crusader-Wars/releases/tag/v1.0.14 and can be found in the `crusader-wars.zip` file within that Crusader-Wars v1.0.14 release.
