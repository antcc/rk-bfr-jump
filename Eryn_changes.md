This change should be made to the source code of Eryn 1.1.9 for our package to work correctly (it does work without it but produces different results):

- **[eryn/moves/group.py, L.133]** The initial condition to setup friends should only be `self.iter == 0`, because
the logic to update the friends every few iterations is already at the end of the method [See [Eryn#19](https://github.com/mikekatz04/Eryn/pull/19)].
