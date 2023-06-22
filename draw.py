# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:40:22 2023

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import string

def fig_multiscale_modeling(no=2, icscale=1):
    
    def subplot1(ax, no=2, icscale=1, border=1):
    
        def circle(ax, center, ocr, icr, colors={"gray":1}):
            cx = center[0]
            cy = center[1]
            random_prob = np.random.uniform(0.0, 1.0)
            prob = 0.0
            for color in colors.keys():
                prob += colors[color]
                if random_prob < prob:
                    ax.add_artist(mpatches.Circle((cx, cy), icr, fill=True, color=color))
                    break
                pass
            pass
        
        area_stats = []
        ocr = float(border) / (no * 2 + 1)
        icr = ocr * 0.1 * icscale
        for x in range(-no, no+1):
            area_stats.append([])
            sx = x * ocr * 2
            for y in range(-no, no+1):
                sy = y * ocr * 2
                ax.add_artist(plt.Rectangle((sx-ocr, sy-ocr), ocr*2, ocr*2, fill=False))
                infected = np.random.uniform(0.0, 1.0) < 0.25
                area_stats[x+no].append(infected)
                pass
            pass
        for i in range((no*2+1)**4):
            ix_random = np.random.uniform(0, 1)
            iy_random = np.random.uniform(0, 1)
            xindex = int(ix_random * (no * 2 + 1))
            yindex = int(iy_random * (no * 2 + 1))
            ix = (ix_random - 0.5) * border * 2
            iy = (iy_random - 0.5) * border * 2
            infected = area_stats[xindex][yindex]
            colors={"blue":1}
            if infected:
                prob = np.random.uniform(0.01, 0.25)
                colors={"red":prob, "blue":1-prob}
                pass
            circle(ax, (ix, iy), ocr, icr, colors)
            pass
        return area_stats, ocr, icr, no, border
    
    def subplot2(ax, area_stats, ocr, icr, no, border):
        
        def circle(ax, center, stats, ocr, icr):
            counts = [0, 0]
            rcx, rcy = center[0], center[1]
            interval = ocr * 2.0 / np.array(stats).shape[0]
            n = int((np.array(stats).shape[0] - 1) / 2)
            for x in range(-n, n+1):
                cx = x * interval + rcx
                for y in range(-no, no+1):
                    cy = y * interval + rcy
                    infected = stats[x+n][y+n]
                    if infected:
                        ax.add_artist(mpatches.Circle((cx, cy), icr, fill=True, color="red"))
                        counts[0] += 1
                        pass
                    else:
                        ax.add_artist(mpatches.Circle((cx, cy), icr, fill=True, color="blue"))
                        counts[1] += 1
                        pass
                    pass
                pass
            return counts
        
        def stats(n, prob):
            stats = []
            for x in range(-n, n+1):
                stats.append([])
                for y in range(-n, n+1):
                    infected = np.random.uniform(0.0, 1.0) < prob
                    stats[x+n].append(infected)
                    pass
                pass
            return stats
        
        area_counts = []
        for x in range(-no, no+1):
            area_counts.append([])
            sx = x * ocr * 2
            for y in range(-no, no+1):
                sy = y * ocr * 2
                if x == 0 and y == 0:
                    ax.add_artist(plt.Rectangle((sx-ocr, sy-ocr), ocr*2, ocr*2, \
                                                fill=True, color="gray", alpha=0.8))
                    counts = circle(ax, (sx, sy), area_stats, ocr, icr)
                    area_counts[x+no].append(counts)
                    pass
                else:
                    ax.add_artist(plt.Rectangle((sx-ocr, sy-ocr), ocr*2, ocr*2, fill=False))
                    n = int((np.array(area_stats).shape[0] - 1) / 2)
                    infected = np.random.uniform(0.0, 1.0) < 0.25
                    prob = 0.0
                    if infected:
                        prob = np.random.uniform(0.01, 0.25)
                        pass
                    sampled_stats = stats(n, prob)
                    counts = circle(ax, (sx, sy), sampled_stats, ocr, icr)
                    area_counts[x+no].append(counts)
                    pass
                pass
            pass
        return area_counts, ocr, icr
    
    def subplot3(ax, area_counts, ocr, icr):
        
        def circles(ax, center, counts, ocr, icr):
            rcx, rcy = center[0], center[1]
            infected, noninfected = counts
            for i in range(noninfected):
                cx = np.random.uniform(rcx-ocr, rcx+ocr)
                cy = np.random.uniform(rcy-ocr, rcy+ocr)
                ax.add_artist(mpatches.Circle((cx, cy), icr, fill=True, color="blue"))
                pass
            for i in range(infected):
                cx = np.random.uniform(rcx-ocr, rcx+ocr)
                cy = np.random.uniform(rcy-ocr, rcy+ocr)
                ax.add_artist(mpatches.Circle((cx, cy), icr, fill=True, color="red"))
                pass
            pass
            
        for x in range(-no, no+1):
            sx = x * ocr * 2
            for y in range(-no, no+1):
                sy = y * ocr * 2
                if x == 0 and  y == 0:
                    ax.add_artist(plt.Rectangle((sx-ocr, sy-ocr), ocr*2, ocr*2, \
                                                fill=True, color="gray", alpha=0.8))
                    pass
                else:
                    ax.add_artist(plt.Rectangle((sx-ocr, sy-ocr), ocr*2, ocr*2, fill=False))
                    pass
                counts = area_counts[x+no][y+no]
                circles(ax, (sx, sy), counts, ocr, icr)
                pass
            pass
        pass
            
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(1, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    area_stats, ocr, icr, no, border = subplot1(ax1, no, icscale, border=1)
    ax1.set(xlim=(-border, border), ylim=(-border, border))
    ax1.axes.yaxis.set_ticks([])
    ax1.axes.xaxis.set_ticks([])
    ax1.set_xlabel('(a)', fontsize=24)
    
    area_counts, ocr, icr = subplot2(ax2, area_stats, ocr, icr, no, border)
    ax2.set(xlim=(-border, border), ylim=(-border, border))
    ax2.axes.yaxis.set_ticks([])
    ax2.axes.xaxis.set_ticks([])
    ax2.set_xlabel('(b)', fontsize=24)
    
    subplot3(ax3, area_counts, ocr, icr)
    ax3.set(xlim=(-border, border), ylim=(-border, border))
    ax3.axes.yaxis.set_ticks([])
    ax3.axes.xaxis.set_ticks([])
    ax3.set_xlabel('(c)', fontsize=24)
    
    ax1_xy1 = [border, border]
    ax1_xy2 = [border, -border]
    ax2_xy1 = [-float(border)/(2*no+1), float(border)/(2*no+1)]
    ax2_xy2 = [-float(border)/(2*no+1), -float(border)/(2*no+1)]
    con1 = mpatches.ConnectionPatch(xyA=ax1_xy1, xyB=ax2_xy1, coordsA="data", coordsB="data", \
                                    axesA=ax1, axesB=ax2, linewidth=1)
    ax2.add_artist(con1)
    con2 = mpatches.ConnectionPatch(xyA=ax1_xy2, xyB=ax2_xy2, coordsA="data", coordsB="data", \
                                    axesA=ax1, axesB=ax2, linewidth=1)
    ax2.add_artist(con2)
    
    ax2_xy1 = [float(border)/(2*no+1), float(border)/(2*no+1)]
    ax2_xy2 = [float(border)/(2*no+1), -float(border)/(2*no+1)]
    ax3_xy1 = [-float(border)/(2*no+1), float(border)/(2*no+1)]
    ax3_xy2 = [-float(border)/(2*no+1), -float(border)/(2*no+1)]
    con3 = mpatches.ConnectionPatch(xyA=ax2_xy1, xyB=ax3_xy1, coordsA="data", coordsB="data", \
                                    axesA=ax2, axesB=ax3, linewidth=1)
    ax3.add_artist(con3)
    con4 = mpatches.ConnectionPatch(xyA=ax2_xy2, xyB=ax3_xy2, coordsA="data", coordsB="data", \
                                    axesA=ax2, axesB=ax3, linewidth=1)
    ax3.add_artist(con4)
    
    pass

def fig_interventions_effect(R0=1.3, Rp=0.9, gamma=1.1, thread=5, tpolicy=35, iteration=70):
    I = 1e-6 * np.ones(5)
    S = 1 - I
    R = 0
    beta = R0 * gamma
    start_t = np.linspace(0, tpolicy*0.5, thread)
    it = []
    for t in range(iteration):
        if t > tpolicy:
            beta = Rp * gamma
            pass
        flag = np.int16(t>start_t)
        i = beta * S * I * flag
        r = gamma * I * flag
        S = S - i
        I = I + i - r
        R = R + r
        it.append(i)
        pass
    plt.figure(figsize=(12, 8))
    plots = plt.plot(it, linewidth=3)
    plt.axvline(tpolicy, color="black", linestyle='--', linewidth=3)
    for x in range(thread):
        color = plots[x].get_color()
        plt.axvline(start_t[x], color=color, linestyle='--', linewidth=3)
        pass
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Number of new infections", fontsize=20)
    plt.text(2+tpolicy, 0.1*np.max(np.array(it)), "Intervention implementation time", fontsize=18)
    plt.text(2+start_t[2], 0.1*np.max(np.array(it)), "Epidemic start time", fontsize=18)
    for x in range(thread):
        plt.plot([start_t[x], 2+start_t[2]], [0, 0.1*np.max(np.array(it))], color="black")
        pass
    plt.plot([tpolicy, 2+tpolicy], [0, 0.1*np.max(np.array(it))], color="black")
    plt.show()
    pass

