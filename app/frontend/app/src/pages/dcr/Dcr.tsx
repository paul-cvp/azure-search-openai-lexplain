import { useRef, useState } from "react";
import ModelerState from "../../components/ModelerState";
import { EventLog, RoleTrace } from "dcr-engine";
import styles from "./Dcr.module.css";

export enum StateEnum {
    Modeler,
    Home,
    Simulator,
    Conformance,
    Discovery,
    EventLogGeneration
}

export interface StateProps {
    savedGraphs: DCRGraphRepository;
    setSavedGraphs: (repository: DCRGraphRepository) => void;
    setState: (state: StateEnum) => void;
    savedLogs: EventLogRepository;
    setSavedLogs: (repository: EventLogRepository) => void;
    lastSavedGraph: React.RefObject<string | undefined>;
    lastSavedLog: React.RefObject<string | undefined>;
}

export type DCRGraphRepository = {
    [name: string]: string;
};

export type EventLogRepository = {
    [name: string]: EventLog<RoleTrace>;
};

export function Component() {
    const [state, setState] = useState(StateEnum.Modeler);
    const [savedGraphs, setSavedGraphs] = useState<DCRGraphRepository>({});
    const [savedLogs, setSavedLogs] = useState<EventLogRepository>({});

    const lastSavedGraph = useRef<string>(undefined);
    const lastSavedLog = useRef<string>(undefined);
    return (
        <div className={styles.dcrContainer}>
            <ModelerState
                savedLogs={savedLogs}
                setSavedLogs={setSavedLogs}
                savedGraphs={savedGraphs}
                setSavedGraphs={setSavedGraphs}
                setState={setState}
                lastSavedGraph={lastSavedGraph}
                lastSavedLog={lastSavedLog}
            />
        </div>
    );
}

Component.displayName = "Dcr";
