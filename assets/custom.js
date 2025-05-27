function setupVideoSeekObserver() {
    const video = document.querySelector("video");
    const target = document.getElementById("video-timestamp");

    if (!video || !target) {
        console.log("⏳ Waiting for video and timestamp element...");
        return false;
    }

    const observer = new MutationObserver(() => {
        try {
            const content = JSON.parse(target.textContent);
            if (content && typeof content.time === "number") {
                console.log("⏩ Seeking to time:", content.time);
                video.currentTime = content.time;
                video.play();
            }
        } catch (err) {
            console.error("❌ Failed to parse timestamp JSON:", err);
        }
    });

    observer.observe(target, { childList: true, characterData: true, subtree: true });
    return true;
}

function setupObserver() {
    const video = document.querySelector("video");
    const target = document.getElementById("video-timestamp");

    if (!video || !target) {
        console.log("⏳ Waiting for video and target...");
        return false;
    }

    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            if (mutation.type === "characterData" || mutation.type === "childList") {
                try {
                    const content = JSON.parse(target.textContent);
                    if (content && content.time !== undefined) {
                        console.log("⏩ Jumping to time:", content.time);
                        video.currentTime = content.time;
                        video.play();
                    }
                } catch (e) {
                    console.error("❌ Failed to parse timestamp JSON:", e);
                }
            }
        }
    });

    // IMPORTANT: Observe both character data and child list!
    observer.observe(target, {
        childList: true,
        subtree: true,
        characterData: true
    });

    console.log("✅ Observer attached");
    return true;
}

const interval = setInterval(() => {
    if (setupObserver()) {
        clearInterval(interval);
    }
}, 200);

