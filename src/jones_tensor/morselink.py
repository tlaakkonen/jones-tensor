from dataclasses import dataclass
from typing import Self

class MorseEvent: pass

@dataclass
class MorseCap(MorseEvent):
    """
    A cap in a MorseLink presentation.
    
    Attributes:
        idx: The first strand which is consumed by the cap.
        down: Whether the orientation is from the first to the second strand or vice-versa.
    """
    idx: int
    down: bool

@dataclass
class MorseCup(MorseEvent):
    """
    A cup in a MorseLink presentation.
    
    Attributes:
        idx: The first strand which is produced by the cup.
        down: Whether the orientation is from the first to the second strand or vice-versa.
    """
    idx: int
    down: bool

@dataclass
class MorseX(MorseEvent):
    """
    A crossing in a MorseLink presentation.
    
    Attributes:
        idx: The first strand which is involved in the crossing.
        over: Whether the first strand crosses over or under the second strand.
        forward_a: Whether the first strand is oriented forward or backward in time.
        forward_b: Whether the second strand is oriented forward or backward in time.
    """

    idx: int
    over: bool
    forward_a: bool
    forward_b: bool

    @property
    def writhe(self):
        """
        Compute whether this crossing is over or under, taking orientation into account
        """
        return +1 if self.over ^ self.forward_a ^ self.forward_b else -1

@dataclass
class MorseLink:
    """
    A presentation of a knot or link in terms of caps, cups, crossings, and monotone planar wires between them.

    It is given as a sequence of 'events' which represent the creation (cups), annihilation (caps), or crossing of a pair of oriented strands.

    Attributes:
        events (list[MorseEvent]): A list of events ordered chronologically.
    """

    events: list[MorseEvent]

    @staticmethod
    def from_pd(pd: list[list[int]]) -> Self:
        """
        Calculate a MorseLink presentation of a link specified by a planar diagram code.

        This code was adapted from the KnotTheory Mathematica package, specifically the
        KnotTheory/src/MorseLink.m module, written by Siddarth Sankaran.

        Parameters:
            pd: A planar diagram presentation of a link, given as a list of 4-tuples of integers.
        """

        events = []
        crossings = list(pd).copy()

        if len(crossings) == 0:
            raise ValueError("at least one crossing is required")

        # First we want to determine the segments assigned to each component
        all_segs = set(sum((list(c) for c in crossings), start=[]))
        comps = []
        while len(all_segs) > 0:
            # Pick a segment and follow it:
            seen = set()
            current = next(iter(all_segs))
            seen.add(current)
            all_segs.remove(current)
            while True:
                nexts = [c[(c.index(current) + 2) % 4] for c in crossings if current in c]
                nexts = [n for n in nexts if n not in seen]
                if len(nexts) == 0: break
                current = nexts[0]
                seen.add(current)
                all_segs.remove(current)
            comps.append(sorted(seen))
        # From this find the successor of each segment in a way that handles wraparound correctly
        nexts = {}
        for comp in comps:
            for i, c in enumerate(comp):
                nexts[c] = comp[(i + 1) % len(comp)]
        # Now we can define an orientation on segments
        is_dir_forward = lambda a, b: nexts[a] == b

        # Handle the initial crossing
        c, *crossings = crossings
        d1, d2 = is_dir_forward(c[0], c[2]), is_dir_forward(c[1], c[3])
        
        events.append(MorseCup(0, d1))
        events.append(MorseCup(2, not d2))
        events.append(MorseX(1, False, d1, d2))
        strands = [c[0], c[3], c[2], c[1]]
        directions = [not d1, d2, d1, not d2]    

        while len(strands) > 0:
            # Try to find a pair of adjacent strands to put a cap:
            found = False
            for i in range(len(strands) - 1):
                if strands[i] != strands[i + 1]: continue
                
                events.append(MorseCap(i, directions[i]))

                del strands[i:i+2]
                del directions[i:i+2]

                found = True
                break
            if found: continue

            # Try to find a crossing that applies to adjacent strands:
            found = False
            for i in range(len(strands) - 1):
                # Look for a crossing for which the strands (i, i + 1) form one half
                for j, c in enumerate(crossings):
                    if strands[i] not in c: continue
                    n = c.index(strands[i])
                    if strands[i + 1] != c[(n + 1) % 4]: continue
                    break
                else:
                    continue

                x, y, b, a = [c[(n + i) % 4] for i in range(4)]
                dx = directions[i]
                dy = directions[i + 1]
                events.append(MorseX(i, n % 2 == 1, dx, dy))

                strands[i:i+2] = [a, b]
                directions[i:i+2] = [dy, dx]
                del crossings[j]

                found = True
                break
            if found: continue

            # Otherwise, add a cup to make a crossing possible:
            for i in range(len(strands) - 1):
                if strands.count(strands[i]) > 1: continue
                # Look for a crossing which contains strands[i], but strands does not contain its partner
                for j, c in enumerate(crossings):
                    if strands[i] not in c: continue
                    n = c.index(strands[i])
                    if c[(n + 1) % 4] in strands: continue
                    break
                else:
                    continue

                x, y, b, a = [c[(n + i) % 4] for i in range(4)]
                dx = directions[i]
                dy = is_dir_forward(y, a)
                events.append(MorseCup(i + 1, not dy))
                events.append(MorseX(i, n % 2 == 1, dx, dy))

                strands[i:i+1] = [a, b, y]
                directions[i:i+1] = [dy, dx, not dy]
                del crossings[j]

                break
            else:
                # If we made no progress, then fail            
                raise RuntimeError("failed to make progress")

        # If we are done, return
        return MorseLink(events)
    
    @staticmethod
    def from_braid(word: list[int], strands: int = None) -> Self:
        """
        Convert a Markov-closed braid into a MorseLink presentation.

        Parameters:
            word: the generators of the braid encoded as non-zero integers
            strands: the number of strands to use, can be usually inferred from the generators
        """
        if strands is None:
            strands = max(abs(g) + 1 for g in word)

        events = []
        for i in range(strands):
            events.append(MorseCup(i, False))

        for gen in word:
            idx = abs(gen) - 1
            events.append(MorseX(idx, gen > 0, True, True))
        
        for i in range(strands - 1, -1, -1):
            events.append(MorseCap(i, True))

        return MorseLink(events)


    def draw(self, unoriented=False):
        """Draw this MorseLink using discopy."""
        from discopy.ribbon import Diagram, Ty, Id, Cap, Cup, Braid

        def id(s): 
            if len(s) == 0:
                return Id(Ty())
            else:
                return Id(s[0].tensor(*s[1:]))

        class TyUp(Ty):
            def __init__(self, unoriented): 
                self.unoriented = unoriented
                super().__init__('↑' if not unoriented else '')

            @property
            def r(self): return TyDown(self.unoriented)
            
        class TyDown(Ty):
            def __init__(self, unoriented): 
                self.unoriented = unoriented
                super().__init__('↓' if not unoriented else '')
            
            @property
            def r(self): return TyUp(self.unoriented)

        t_up, t_down = TyUp(unoriented), TyDown(unoriented)
        diag = Diagram(inside=(), dom=Ty(), cod=Ty())
        strands = []
        for event in self.events:
            layer = None
            if isinstance(event, MorseCup):
                inner = Cap(t_up if event.down else t_down, t_down if event.down else t_up)
                layer = id(strands[:event.idx]) @ inner @ id(strands[event.idx:])
                strands.insert(event.idx, t_up if event.down else t_down)
                strands.insert(event.idx + 1, t_down if event.down else t_up)
            elif isinstance(event, MorseCap):
                inner = Cup(t_down if event.down else t_up, t_up if event.down else t_down)
                layer = id(strands[:event.idx]) @ inner @ id(strands[event.idx+2:])
                del strands[event.idx + 1]
                del strands[event.idx]
            elif isinstance(event, MorseX):
                inner = Braid(t_down if event.forward_a else t_up, t_down if event.forward_b else t_up, event.over)
                layer = id(strands[:event.idx]) @ inner @ id(strands[event.idx+2:])
                strands[event.idx], strands[event.idx+1] = strands[event.idx+1], strands[event.idx]
            diag = diag >> layer
        
        diag.draw()

    def writhe(self):
        """Compute the writhe of this link, taking orientation into account."""
        return sum(c.writhe for c in self.events if isinstance(c, MorseX))
